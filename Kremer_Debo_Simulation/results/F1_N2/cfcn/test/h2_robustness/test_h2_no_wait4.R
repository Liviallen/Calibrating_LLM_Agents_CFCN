.libPaths(c('/tmp/h2-r-library', .libPaths()))
library(brglm2)
args <- commandArgs(trailingOnly=TRUE)
base <- normalizePath(args[1])
out <- file.path(base,'test','h2_robustness')
set.seed(20261024)
B <- 1999L
results <- list()
for (q in c('Q50')) {
 d <- read.csv(file.path(base,'validation',paste0(q,'_Q00_clean.csv')))
 d <- d[d$subject_type==0 & d$wait_time<4,]
 d$treat <- as.integer(d$treatment_name==q)
 # Each validation file comprises two equal batches of independently initialized cohorts.
 n_total <- if(q=='Q05') 364L else 442L
 d$batch <- ifelse(d$treat==1, 1L+(d$cohort_id>n_total/2),1L+(d$cohort_id>n_total+104L))
 cells <- unique(d[c('treat','wait_time')]); cells <- cells[order(cells$treat,cells$wait_time),]
 X <- model.matrix(~wait_time*treat,cells)
 cluster <- unique(d[c('cohort_id','treat','batch')])
 succ <- fail <- matrix(0,nrow(cluster),nrow(cells))
 for(i in seq_len(nrow(d))) {
  a <- match(d$cohort_id[i],cluster$cohort_id)
  b <- which(cells$treat==d$treat[i] & cells$wait_time==d$wait_time[i])
  if(d$subject_choice[i]==1) succ[a,b] <- succ[a,b]+1 else fail[a,b] <- fail[a,b]+1
 }
 fit_counts <- function(s,f) {
  keep <- (s+f)>0
  z <- suppressWarnings(brglmFit(X[keep,,drop=FALSE],s[keep]/(s[keep]+f[keep]),weights=(s+f)[keep],family=binomial('probit'),control=brglmControl(type='AS_mean',maxit=300)))
  if(!isTRUE(z$converged)) return(NA_real_)
  unname(z$coefficients['wait_time:treat'])
 }
 estimate <- fit_counts(colSums(succ),colSums(fail))
 strata <- split(seq_len(nrow(cluster)), interaction(cluster$treat,cluster$batch,drop=TRUE))
 boot <- numeric(B)
 for(b in seq_len(B)) {
  ids <- unlist(lapply(strata,function(v) sample(v,length(v),replace=TRUE)),use.names=FALSE)
  boot[b] <- tryCatch(fit_counts(colSums(succ[ids,,drop=FALSE]),colSums(fail[ids,,drop=FALSE])),error=function(e) NA_real_)
  if(b %% 500==0) {cat(q,b,'bootstrap fits completed\n');flush.console()}
 }
 good <- boot[is.finite(boot)]
 ci <- quantile(good,c(.025,.975),names=FALSE)
 # Centered, non-studentized bootstrap approximation to a two-sided null test.
 p <- (1+sum(abs(good-estimate)>=abs(estimate)))/(length(good)+1)
 results[[q]] <- data.frame(comparison=paste0(q,' vs Q00'),method='Mean-bias-reduced probit; stratified cohort bootstrap',estimate=estimate,ci_low=ci[1],ci_high=ci[2],p_two_sided_centered_bootstrap=p,bootstrap_success=length(good),bootstrap_requested=B,n=nrow(d),clusters=nrow(cluster))
 write.csv(data.frame(interaction=boot),file.path(out,paste0(q,'_no_wait4_probit_bootstrap.csv')),row.names=FALSE)
 print(results[[q]])
}
r <- do.call(rbind,results)
r$p_holm <- p.adjust(r$p_two_sided_centered_bootstrap,'holm')
write.csv(r,file.path(out,'no_wait4_probit_results.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'no_wait4_R_session_info.txt'))
