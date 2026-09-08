# ============================================================================
# Two-state HMM for American Redstarts
# Motus detections -> clean/filter data -> add hourly weather ->
# fit 2-state HMM -> identify Resting/Active -> calculate bouts/occupancy ->
# predict transition probabilities -> save
# ============================================================================

library(nanoparquet)
library(dplyr)
library(tidyr)
library(readxl)
library(writexl)
library(hmmTMB)
library(ggplot2)

setwd("C:/Users/goswa/OneDrive/Desktop/hmm output")

YEAR   <- 2024  # year of interest
OFFSET <- 0.001   # keep exact-zero SDs strictly positive for the gamma
TZ     <- "America/Jamaica"
#set.seed(20250820) #if you want the same randomness every time
CI_LEVEL <- 0.95
N_POST   <- 1000

WEATHER_FILE <- "C:/Users/goswa/Downloads/weather correlation 1/jamaica_era5_with_daily.xlsx"
OUTDIR       <- sprintf("%d redstart hmm new code test", YEAR)

# create folder if it doesn't already exist
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
op <- function(fmt, ...) file.path(OUTDIR, sprintf(fmt, ...))

# protect against crashing and salvage what already fit
safe <- function(expr, default = NA) tryCatch(expr, error = function(e) default)

#Read prepared hmm dataset
year_tag  <- paste0("y", YEAR)
DATA_FILE <- "C:/Users/goswa/OneDrive/Desktop/FINAL-motus-processing-and-hmm-dataset_final/hmm_dataset.parquet"
if (!file.exists(DATA_FILE)) stop("Detections file not found: ", DATA_FILE)

raw <- read_parquet(DATA_FILE) %>% as.data.frame()

#make sure columns are as expected 
cat("\nColumns in the detections file:\n"); print(names(raw))

if (!"sig_sd_db" %in% names(raw)) stop("sig_sd_db not found in ", DATA_FILE)
raw$sd_sig <- raw$sig_sd_db + OFFSET

need <- c("local_time", "date", "period", "tag_id", "track_id", "doy",
          "tsCorrected", "age", "sex")
miss <- setdiff(need, names(raw))
if (length(miss)) stop("Missing expected column(s): ", paste(miss, collapse = ", "))

# correct, local `date` and `doy` come from the hmm dataset 
# do NOT change them and be very careful that weather, R timestamp, and hmmdataset times are all correct + consistent!!!!!
dat <- raw %>%
  mutate(time = as.POSIXct(local_time, tz = TZ),
         date = as.Date(time, tz = TZ)) %>%
  filter(as.integer(format(date, "%Y")) == YEAR,
         tolower(period) == "day") %>%
  mutate(ID = track_id)

if (nrow(dat) == 0) stop("No daytime detections in year ", YEAR)
cat(sprintf("\nYear: %d | daytime rows: %d | days spanned: %d\n",
            YEAR, nrow(dat), length(unique(dat$date))))

# drop tags with missing age/sex, they'll be bad for the model
n_pre_meta <- nrow(dat)
dat <- dat %>% filter(!is.na(age), !is.na(sex))

#warning if necessary before continuing 
if (nrow(dat) == 0)
  stop("Every detection in ", YEAR, " lacks age or sex (", n_pre_meta,
       " rows dropped). Backfill the metadata before fitting this year.")
cat(sprintf("Birds with age and sex: %d\n", length(unique(dat$tag_id))))

# Join hourly weather data to hmm dataset
# Both files are ALREADY local time
dat <- dat %>%
  mutate(hour     = as.integer(format(time, "%H")),
         join_key = format(time, "%Y-%m-%d %H"))

# tiemstamp double check 
stopifnot(all(as.Date(dat$time, tz = TZ) == dat$date))
cat("hour range: ", paste(range(dat$hour), collapse = "-"), "\n")

wx <- read_excel(WEATHER_FILE, sheet = "hourly") %>% as.data.frame()

#MUST BE LOCAL TIME TO JOIN 
if (!"datetime_local" %in% names(wx))
  stop("Column 'datetime_local' not found in the hourly weather sheet.")

# make sure time label is still correct local time 
if (is.character(wx$datetime_local)) {
  wx$datetime_local <- as.POSIXct(wx$datetime_local, tz = TZ)
} else {
  wx$datetime_local <- as.POSIXct(
    format(wx$datetime_local, "%Y-%m-%d %H:%M:%S", tz = "UTC"), tz = TZ)
}

#precipitation units check 
# weather file was converted  to mm 
p_max_raw <- max(wx$precipitation, na.rm = TRUE)
if (p_max_raw < 0.1) {
  wx$precipitation <- wx$precipitation * 1000
  cat(sprintf("Precipitation: max raw value %.6g -> assumed METRES, converted to mm (new max %.4g).\n",
              p_max_raw, max(wx$precipitation, na.rm = TRUE)))
} else {
  cat(sprintf("Precipitation: max raw value %.4g -> assumed already in mm, left unchanged.\n",
              p_max_raw))
}

#match weather to hmm dataset, make sure everything is lined up 

wx <- wx %>%
  mutate(join_key = format(datetime_local, "%Y-%m-%d %H")) %>%
  select(join_key, temperature, precipitation) %>%
  filter(!is.na(join_key)) %>%
  distinct(join_key, .keep_all = TRUE)

n_before  <- nrow(dat)
dat       <- dat %>% left_join(wx, by = "join_key")
unmatched <- with(dat, is.na(temperature) | is.na(precipitation))
cat(sprintf("\nWeather join: %d detections in, %d unmatched (%.2f%%), %d retained.\n",
            n_before, sum(unmatched), 100 * mean(unmatched), sum(!unmatched)))
if (sum(unmatched) > 0) {
  bad_hours <- sort(unique(dat$join_key[unmatched]))
  cat(sprintf("Distinct local hours with no weather row: %d (first few: %s)\n",
              length(bad_hours), paste(head(bad_hours, 5), collapse = ", ")))
}
dat <- dat[!unmatched, , drop = FALSE]
if (nrow(dat) == 0) stop("No detections survived the weather join.")


#weather data is now matched to hmm dataset 

#################################################################################################################

#more cleaning of combined data 

# Single-observation tracks are dropped. some tracks may become single-observation if they dont have enough weather data 

AGE_LEVELS <- c("ASY", "SY")   # reference = ASY (after second year)
SEX_LEVELS <- c("F", "M")      # reference = F
#you can change the reference values if you want 

dat <- dat %>%
  group_by(track_id) %>% filter(n() >= 2) %>% ungroup() %>% #track must have at least two detections
  mutate(tag = factor(tag_id),
         age = factor(age, levels = AGE_LEVELS),
         sex = factor(sex, levels = SEX_LEVELS),
         doy = as.integer(doy)) %>%          
  arrange(track_id, tsCorrected) %>%
  as.data.frame()

# age and sex must match what you set earlier, other values shouldn't be present 
if (anyNA(dat$age) || anyNA(dat$sex)) {
  stop("Labels outside AGE_LEVELS/SEX_LEVELS found. Observed values were:\n",
       "  age: ", paste(sort(unique(raw$age)), collapse = ", "), "\n",
       "  sex: ", paste(sort(unique(raw$sex)), collapse = ", "), "\n",
       "Update AGE_LEVELS / SEX_LEVELS to match.")
}

#reference from what was set earlier. You can change it here if you want
cat(sprintf("Reference class: age = %s, sex = %s\n", AGE_LEVELS[1], SEX_LEVELS[1]))
cat("Birds per age x sex cell:\n")
print(table(dat %>% distinct(tag, age, sex) %>% pull(age),
            dat %>% distinct(tag, age, sex) %>% pull(sex)))

# stored orthonormal bases 
doy_basis  <- poly(dat$doy,  2)
hour_basis <- poly(dat$hour, 2)
dat$doy_p1  <- doy_basis[, 1];  dat$doy_p2  <- doy_basis[, 2]
dat$hour_p1 <- hour_basis[, 1]; dat$hour_p2 <- hour_basis[, 2]

# persist the bases so prediction grids can be put on the SAME scale later.
# poly() is orthonormal to THIS sample, so a grid built with a fresh poly()
# call would not match the fitted coefficients.
poly_bases <- list(
  doy  = list(coefs = attr(doy_basis,  "coefs"), degree = 2,
              observed_range = range(dat$doy)),
  hour = list(coefs = attr(hour_basis, "coefs"), degree = 2,
              observed_range = range(dat$hour)),
  year = YEAR, n_rows = nrow(dat)
)
saveRDS(poly_bases, op("poly_bases_%s.rds", year_tag))


#########################################################
# FIT THE MODEL 

#transition covariates 

FORM <- ~ age + sex + doy_p1 + doy_p2 + hour_p1 + hour_p2 +
  temperature + precipitation + s(tag, bs = "re")

# specifiy number of hidden states

hid <- MarkovChain$new(data = dat, n_states = 2,
                       formula = FORM,
                       initial_state = "stationary") #because initial state is unknown 

#observation's distributions and guess values 
obs <- Observation$new(
  data     = dat,
  dist     = list(sd_sig = "gamma2"), #gamma distribution for sd of signal 
  n_states = 2,
  par      = list(sd_sig = list(mean = c(0.3, 4.5), sd = c(0.4, 3))) #'guess' values for active and rest sd of sig 
)

#run the model 
mod <- HMM$new(obs = obs, hid = hid)

cat("Fitting...\n")
t0 <- Sys.time()
mod$fit(silent = TRUE) # If you're suspicious it won't converge you can change to FALSE 
t1 <- Sys.time()
runtime_min <- as.numeric(difftime(t1, t0, units = "mins"))
cat(sprintf("Fit finished in %.1f min.\n", runtime_min))




#Decode states by mean SD of sig 

par2       <- mod$obs()$par()[, , 1]
rest_state <- which.min(par2["sd_sig.mean", ])   # low SD = Resting, higher SD = active 
act_state  <- setdiff(1:2, rest_state)

dat$state_raw <- mod$viterbi()
dat$behaviour <- factor(ifelse(dat$state_raw == rest_state, "Resting", "Active"),
                        levels = c("Resting", "Active"))
dat$p_active  <- mod$state_probs()[, act_state]



# Bout lengths 
# computed WITHIN each bird-day track, so no bout spans a tracks with a break between them 
bout_list <- lapply(split(seq_len(nrow(dat)), dat$ID), function(idx) {
  d <- dat[idx, , drop = FALSE]
  d <- d[order(d$time), , drop = FALSE]
  r <- rle(as.character(d$behaviour))
  ends   <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1
  data.frame(ID          = d$ID[1],
             tag         = as.character(d$tag[1]),
             date        = d$date[1],
             state       = r$values,
             n_obs       = r$lengths,
             elapsed_min = as.numeric(difftime(d$time[ends], d$time[starts],
                                               units = "mins")),
             stringsAsFactors = FALSE)
})
bouts <- do.call(rbind, bout_list)
rownames(bouts) <- NULL
bouts$state <- factor(bouts$state, levels = c("Resting", "Active"))

bout_summary <- bouts %>%
  group_by(state) %>%
  summarise(n_bouts        = n(),
            mean_min       = mean(n_obs),
            median_min     = median(n_obs),
            sd_min         = sd(n_obs),
            max_min        = max(n_obs),
            mean_elapsed   = mean(elapsed_min),
            median_elapsed = median(elapsed_min),
            max_elapsed    = max(elapsed_min),
            .groups = "drop")

bout_bins <- c(0, 1, 2, 3, 5, 10, 20, 30, 60, 120, Inf)
bout_dist <- bouts %>%
  mutate(bin = cut(n_obs, breaks = bout_bins, right = TRUE,
                   labels = c("1", "2", "3", "4-5", "6-10", "11-20",
                              "21-30", "31-60", "61-120", ">120"))) %>%
  count(state, bin) %>%
  group_by(state) %>% mutate(pct = 100 * n / sum(n)) %>% ungroup() %>%
  as.data.frame()

#Summaries and occupancy tables 


conv <- safe(mod$out()$convergence) # did the model converge successfully 
nll  <- safe(as.numeric(mod$out()$objective)) #save things even if something breaks later 
k    <- safe(length(mod$out()$par))
if (!is.na(nll) && !is.na(k)) {
  aic <- 2 * nll + 2 * k
  bic <- 2 * nll + k * log(nrow(dat))
} else {
  aic <- bic <- NA
}

tpm_all <- safe(mod$hid()$tpm(), NULL)
occ     <- table(dat$behaviour)

occ_bird <- dat %>%
  count(tag, behaviour) %>%
  group_by(tag) %>%
  mutate(pct = round(100 * n / sum(n), 2)) %>%
  ungroup() %>%
  pivot_wider(names_from = behaviour, values_from = c(n, pct), values_fill = 0) %>%
  as.data.frame()

sum_file <- op("amre_%s_hour_weather_summary.txt", year_tag) 
con <- file(sum_file, open = "wt")
sink(con, split = TRUE)

# make sure you save everything even if a print fails later 
tryCatch({
  
  cat(sprintf("== AMRE %d daytime | all birds ==\n", YEAR))
  cat("Transition model: age + sex + poly(doy,2) + poly(hour,2)\n")
  cat("                  + temperature + precipitation (raw)\n")
  cat("                  + s(tag, bs = 're')\n\n")
  cat(sprintf("Year         : %d\n", YEAR))
  cat(sprintf("Days spanned : %d\n", length(unique(dat$date))))
  cat(sprintf("Birds        : %d\n", nlevels(droplevels(dat$tag))))
  cat(sprintf("Detections   : %d\n", nrow(dat)))
  cat(sprintf("Bird-days    : %d\n", length(unique(dat$ID))))
  cat(sprintf("Runtime      : %.1f min\n", runtime_min))
  
  cat(sprintf("\nConvergence code: %s (%s)\n", conv,
              if (identical(conv, 0L) || identical(conv, 0)) "converged" else "CHECK THIS"))
  cat(sprintf("nll = %.2f   k = %s   AIC = %.2f   BIC = %.2f\n", nll, k, aic, bic))
  
  cat("\n== Transition-model coefficients ==\n")
  print(safe(mod$coeff_fe()))
  
  cat("\n== Gamma emission on SD (dB) ==\n"); print(round(par2, 3))
  cat(sprintf("\nState %d = Resting, State %d = Active\n", rest_state, act_state))
  
  cat("\n== Transition matrix at first row's covariate values (random effects at 0) ==\n")
  print(round(tpm_all[, , 1], 4))
  
  cat("\n== Occupancy (Viterbi) ==\n")
  print(occ)
  cat("\nPercent of detections:\n")
  print(round(100 * prop.table(occ), 2))
  
  cat("\n== Occupancy per bird ==\n")
  print(occ_bird)
  
  cat("\n== Bout lengths (Viterbi runs, within bird-day tracks) ==\n")
  cat("n_obs = number of ~1-min detection windows in the run.\n")
  cat("elapsed = wall-clock minutes from first to last detection in the run\n")
  cat("          (smaller than n_obs would suggest only if windows are gapped).\n\n")
  print(as.data.frame(bout_summary), digits = 4)
  
  cat("\n== Bout-length distribution (n_obs bins) ==\n")
  print(bout_dist, digits = 4)
  
  cat("\n== Expected dwell time implied by the fitted TPM ==\n")
  dwell_first <- 1 / (1 - diag(tpm_all[, , 1]))
  cat("At the first row's covariate values:\n")
  cat(sprintf("  Resting: %.2f min    Active: %.2f min\n",
              dwell_first[rest_state], dwell_first[act_state]))
  if (dim(tpm_all)[3] > 1) {
    mean_diag  <- c(mean(tpm_all[1, 1, ]), mean(tpm_all[2, 2, ]))
    dwell_mean <- 1 / (1 - mean_diag)
    cat("Averaged over all rows' TPMs:\n")
    cat(sprintf("  Resting: %.2f min    Active: %.2f min\n",
                dwell_mean[rest_state], dwell_mean[act_state]))
  }
  cat("\nCompare with the empirical means above. A large gap between the two is a\n")
  cat("sign the geometric dwell-time assumption is being strained.\n")
  
}, finally = { sink(); close(con) })


# ---- 8. Save ---------------------------------------------------------------
saveRDS(mod, op("hmm_amre_%s_hour_weather.rds", year_tag))

# decoded states go to CSV probably too big for Excel 

dat_out <- dat %>% mutate(across(where(is.factor), as.character))
write.csv(dat_out, op("amre_%s_decoded.csv", year_tag), row.names = FALSE)
#############################################################################################################



#confidence intervals coefficients and signficance tests - wald 

ci_all <- safe(mod$confint(level = CI_LEVEL), NULL) #mod$confint
transition_ci <- NULL
joint_wald    <- NULL

if (is.null(ci_all) || is.null(ci_all$coeff_fe$hid)) {
  cat("\nconfint() unavailable - skipping coefficient CIs and joint tests.\n")
} else {
  
  hid_ci <- as.data.frame(ci_all$coeff_fe$hid)
  hid_ci$parameter <- rownames(as.data.frame(ci_all$coeff_fe$hid))
  rownames(hid_ci) <- NULL
  
  if (!all(c("mle", "lcl", "ucl") %in% names(hid_ci)))
    stop("Unexpected confint() columns: ", paste(names(hid_ci), collapse = ", "))
  
  zc <- qnorm(1 - (1 - CI_LEVEL) / 2)
  if (!"se" %in% names(hid_ci)) hid_ci$se <- (hid_ci$ucl - hid_ci$lcl) / (2 * zc)
  
  hid_ci$wald_z   <- hid_ci$mle / hid_ci$se
  hid_ci$p_value  <- 2 * pnorm(abs(hid_ci$wald_z), lower.tail = FALSE)
  hid_ci$evidence <- ifelse(hid_ci$lcl > 0 | hid_ci$ucl < 0,
                            "Evidence of effect", "No clear evidence")
  
  # change model labels to our labels rest and active based on sd
  ra_prefix <- sprintf("S%d>S%d", rest_state, act_state)
  ar_prefix <- sprintf("S%d>S%d", act_state, rest_state)
  hid_ci$transition <- ifelse(grepl(paste0("^", ra_prefix, "\\."), hid_ci$parameter),
                              "Rest -> Active",
                              ifelse(grepl(paste0("^", ar_prefix, "\\."), hid_ci$parameter),
                                     "Active -> Rest", NA_character_))
  
  # with 2 states each row has one off-diagonal, so the sign of the
  # coefficient maps directly onto that transition probability
  hid_ci$direction <- ifelse(hid_ci$lcl > 0 | hid_ci$ucl < 0,
                             ifelse(hid_ci$mle > 0, "increases", "decreases"), "-")
  
  transition_ci <- hid_ci[grepl("^S[12]>S[12]\\.", hid_ci$parameter), ]
  
  write.csv(transition_ci, op("amre_%s_transition_coefficients.csv", year_tag),
            row.names = FALSE)
  cat("\n== Transition coefficients with 95% CI ==\n")
  print(transition_ci[, c("parameter", "transition", "mle", "se",
                          "lcl", "ucl", "p_value", "evidence", "direction")],
        digits = 4)
  
  #joint tests for the polynomial pairs 
  
  
  rep_sd <- safe(mod$tmb_rep(), NULL)
  if (is.null(rep_sd)) rep_sd <- safe(TMB::sdreport(mod$tmb_obj()), NULL)
  
  if (is.null(rep_sd) || is.null(rep_sd$cov.fixed)) {
    cat("\nNo sdreport covariance - skipping joint tests.\n")
  } else {
    
    idx   <- which(names(rep_sd$par.fixed) == "coeff_fe_hid")
    cf    <- as.matrix(mod$hid()$coeff_fe())
    nm    <- rownames(cf)
    b_all <- as.vector(cf)
    
    if (length(idx) != length(b_all))
      stop("coeff_fe() has ", length(b_all), " entries but cov.fixed has ",
           length(idx), " coeff_fe_hid rows - do not trust the pairing.")
    
    V_hid <- rep_sd$cov.fixed[idx, idx, drop = FALSE]
    
    # cross-check: these SEs should match confint()'s for the same parameters
    se_chk <- sqrt(diag(V_hid))
    m <- match(nm, transition_ci$parameter)
    if (!any(is.na(m)) &&
        !isTRUE(all.equal(unname(se_chk), unname(transition_ci$se[m]), tolerance = 1e-4)))
      warning("Covariance-block SEs do not match confint() SEs - check ordering.")
    
    joint_one <- function(terms, prefix, label, trans_label) {
      ii <- grep(paste0("^", prefix, "\\.(", paste(terms, collapse = "|"), ")$"), nm)
      if (length(ii) != length(terms)) return(NULL)
      bb <- b_all[ii]
      W  <- as.numeric(t(bb) %*% solve(V_hid[ii, ii, drop = FALSE], bb))
      data.frame(test = label, transition = trans_label,
                 terms = paste(nm[ii], collapse = " + "),
                 df = length(ii), chisq = W,
                 p_value = pchisq(W, df = length(ii), lower.tail = FALSE),
                 evidence = ifelse(W > qchisq(CI_LEVEL, length(ii)),
                                   "Evidence of overall effect", "No clear evidence"),
                 stringsAsFactors = FALSE)
    }
    
    joint_wald <- do.call(rbind, list(
      joint_one(c("doy_p1","doy_p2"),   ra_prefix, "Day of year", "Rest -> Active"),
      joint_one(c("doy_p1","doy_p2"),   ar_prefix, "Day of year", "Active -> Rest"),
      joint_one(c("hour_p1","hour_p2"), ra_prefix, "Hour of day", "Rest -> Active"),
      joint_one(c("hour_p1","hour_p2"), ar_prefix, "Hour of day", "Active -> Rest")))
    
    if (!is.null(joint_wald)) {
      write.csv(joint_wald, op("amre_%s_joint_wald_tests.csv", year_tag),
                row.names = FALSE)
      cat("\n== Joint Wald tests (2 df each) ==\n"); print(joint_wald, digits = 4)
    }
  }
}



# TPMs
# One covariate varied at a time over its range, 
#everything else held at the reference cell, averaged across birds 
#(each tag's random intercept is
# evaluated and the resulting TPMs are averaged)

GRID_N <- 8

grid_seq <- function(x, n = GRID_N)
  seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = n)

REF <- list(age           = AGE_LEVELS[1], #ref cell 
            sex           = SEX_LEVELS[1], #ref cell 
            doy           = median(dat$doy),
            hour          = median(dat$hour),
            temperature   = median(dat$temperature),
            precipitation = median(dat$precipitation))

cat(sprintf(paste0("\nGrid TPM reference cell: age = %s, sex = %s, doy = %s, ",
                   "hour = %s, temperature = %.2f, precipitation = %.3f\n"),
            REF$age, REF$sex, REF$doy, REF$hour,
            REF$temperature, REF$precipitation))

tag_levels <- levels(droplevels(dat$tag))

tpm_grid_one <- function(varname, values) {
  
  nd <- expand.grid(value = values, tag = tag_levels,
                    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  
  nd$age           <- factor(REF$age, levels = AGE_LEVELS)
  nd$sex           <- factor(REF$sex, levels = SEX_LEVELS)
  nd$doy           <- REF$doy
  nd$hour          <- REF$hour
  nd$temperature   <- REF$temperature
  nd$precipitation <- REF$precipitation
  
  nd[[varname]] <- nd$value
  
  db <- predict(doy_basis,  nd$doy)
  hb <- predict(hour_basis, nd$hour)
  nd$doy_p1  <- db[, 1]; nd$doy_p2  <- db[, 2]
  nd$hour_p1 <- hb[, 1]; nd$hour_p2 <- hb[, 2]
  
  nd$tag    <- factor(nd$tag, levels = levels(dat$tag))
  nd$ID     <- dat$ID[1]
  nd$sd_sig <- dat$sd_sig[1]
  
  pr <- mod$predict("tpm", newdata = nd, n_post = N_POST, return_post = TRUE)
  
  n_value <- length(values)
  
  # averaging across birds 
  avg <- function(arr, i, j) rowMeans(matrix(arr[i, j, ], nrow = n_value))
  
  probs <- c((1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2)
  cell <- function(i, j) {
    dr <- vapply(pr$post, function(A) avg(A, i, j), numeric(n_value))
    q  <- apply(dr, 1, quantile, probs = probs, na.rm = TRUE)
    list(est = avg(pr$mle, i, j), lcl = q[1, ], ucl = q[2, ])
  }
  
  rr <- cell(rest_state, rest_state); ra <- cell(rest_state, act_state)
  ar <- cell(act_state,  rest_state); aa <- cell(act_state,  act_state)
  
  data.frame(covariate = varname, value = values,
             rest_rest     = rr$est, rest_rest_lcl     = rr$lcl, rest_rest_ucl     = rr$ucl,
             rest_active   = ra$est, rest_active_lcl   = ra$lcl, rest_active_ucl   = ra$ucl,
             active_rest   = ar$est, active_rest_lcl   = ar$lcl, active_rest_ucl   = ar$ucl,
             active_active = aa$est, active_active_lcl = aa$lcl, active_active_ucl = aa$ucl,
             n_birds = length(tag_levels), stringsAsFactors = FALSE)
}
tpm_grid <- safe(
  bind_rows(
    tpm_grid_one("doy",           grid_seq(dat$doy)),
    tpm_grid_one("hour",          grid_seq(dat$hour)),
    tpm_grid_one("temperature",   grid_seq(dat$temperature)),
    tpm_grid_one("precipitation", grid_seq(dat$precipitation))
  ) %>%
    mutate(dwell_rest   = 1 / (1 - rest_rest),
           dwell_active = 1 / (1 - active_active)) %>%
    as.data.frame(),
  NULL
)

if (is.null(tpm_grid)) {
  cat("\nGrid TPM failed; model and decoded states are still saved.\n")
} else {
  write.csv(tpm_grid, op("amre_%s_transition_matrices.csv", year_tag),
            row.names = FALSE)
  cat("\n== Grid TPM (tag-averaged) ==\n"); print(tpm_grid, digits = 4)
}

#create the summary excel 
write_xlsx(
  list(
    bouts        = bouts,
    bout_summary = as.data.frame(bout_summary),
    bout_dist    = bout_dist,
    occupancy    = occ_bird,
    tpm          = if (is.null(tpm_grid)) data.frame() else tpm_grid,
    transition_coefficients = if (is.null(transition_ci)) data.frame() else transition_ci,
    joint_wald_tests        = if (is.null(joint_wald))    data.frame() else joint_wald
  ),
  op("amre_%s_summaries.xlsx", year_tag)
)
cat(sprintf("\nAll outputs written to: %s\n", normalizePath(OUTDIR)))