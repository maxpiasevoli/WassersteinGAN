library(rstan)
library(ggplot2)
library(reshape2)

setwd("C:/Users/maxpi/Documents/SML310/WassersteinGAN")

# behavioral learning experiment hierarchical models

# read-in data 
y1 <- as.matrix (read.table ("./data/dogs.dat", skip=1), nrows=30, ncol=25)
y <- ifelse (y1[,]=="S",1,0)
n.dogs <- nrow(y)
n.trials <- ncol(y)

# fit model 1
dataList.1 <- list(n_dogs=n.dogs,n_trials=n.trials,y=y)
dogs.sf1 <- stan(file='./stan/dogs.stan', data=dataList.1, iter=1000, chains=4) # model with intercept
post1 <- extract(dogs.sf1)
beta1 <- colMeans(post1$beta)

# generate samples for model 1
n.sims <- 1
n.dogs.sims = 1000
y.rep.m1 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
p.rep.m1 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
for (j in 1:n.dogs.sims){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  for (t in 1:n.trials){  
    p.rep.m1[,j,t] <- plogis (beta1[1] + beta1[2]*n.avoid.rep + beta1[3]*n.shock.rep)
    y.rep.m1[,j,t] <- rbinom (n.sims, 1, p.rep.m1[,j,t])
    n.avoid.rep <- n.avoid.rep + 1 - y.rep.m1[,j,t] 
    n.shock.rep <- n.shock.rep + y.rep.m1[,j,t] 
  }
}
#write.csv(y.rep.m1[1,,], file='./data/bl_m1_1000.csv', row.names=FALSE)

# model 2
dogs.sf2 <- stan(file='./stan/dogs_no_int.stan', data=dataList.1, iter=1000, chains=4) # model with intercept
post2 <- extract(dogs.sf2)
beta2 <- colMeans(post2$beta)

n.sims <- 1
y.rep.m2 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
p.rep.m2 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
for (j in 1:n.dogs.sims){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  for (t in 1:n.trials){  
    p.rep.m2[,j,t] <- exp(beta2[1]*n.avoid.rep + beta2[2]*n.shock.rep)
    y.rep.m2[,j,t] <- rbinom (n.sims, 1, p.rep.m2[,j,t])
    n.avoid.rep <- n.avoid.rep + 1 - y.rep.m2[,j,t] 
    n.shock.rep <- n.shock.rep + y.rep.m2[,j,t] 
  }
}
#write.csv(y.rep.m2[1,,], file='./data/bl_m2_1000.csv', row.names=FALSE)

# model 3
dogs.sf3 <- stan(file='./stan/dogs_linear.stan', data=dataList.1, iter=1000, chains=4) # model with intercept
post3 <- extract(dogs.sf3)
alpha3 <- colMeans(post3$alpha)
gamma3 <- colMeans(post3$gamma)


n.sims <- 1
y.rep.m3 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
p.rep.m3 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
for (j in 1:n.dogs.sims){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  for (t in 1:n.trials){  
    p.rep.m3[,j,t] <- plogis(alpha3/t+gamma3)
    y.rep.m3[,j,t] <- rbinom (n.sims, 1, p.rep.m3[,j,t])
  }
}
#write.csv(y.rep.m3[1,,], file='./data/bl_m3_1000.csv', row.names=FALSE)

# model 4
dogs.sf4 <- stan(file='./stan/dogs_switch.stan', data=dataList.1, iter=1000, chains=4) # model with intercept
post4 <- extract(dogs.sf4)
gamma4 <- colMeans(post4$gamma)

n.sims <- 1
y.rep.m4 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
p.rep.m4 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
for (j in 1:n.dogs.sims){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  gamma.index <- ceiling(runif(1)*30)
  for (t in 1:n.trials){  
    p.rep.m4[,j,t] <- plogis(100 * (gamma4[gamma.index] - t))
    y.rep.m4[,j,t] <- rbinom (n.sims, 1, p.rep.m4[,j,t])
  }
}
#write.csv(y.rep.m4[1,,], file='./data/bl_m4_1000.csv', row.names=FALSE)

# model 5
dogs.sf5 <- stan(file='./stan/dogs_linear_nl.stan', data=dataList.1, iter=1000, chains=4) # model with intercept
post5 <- extract(dogs.sf5)
alpha5 <- colMeans(post5$alpha)

n.sims <- 1
y.rep.m5 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
p.rep.m5 <- array (NA, c(n.sims, n.dogs.sims, n.trials))
for (j in 1:n.dogs.sims){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  for (t in 1:n.trials){  
    p.rep.m5[,j,t] <- alpha5/t
    y.rep.m5[,j,t] <- rbinom (n.sims, 1, p.rep.m5[,j,t])
  }
}
#write.csv(y.rep.m5[1,,], file='./data/bl_m5_1000.csv', row.names=FALSE)

# stop and frisk models
n.ep <- as.data.frame(read.csv("./data/2014_arrests_weapon.csv", sep=","))
y.ep <- as.data.frame(read.csv("./data/20152016_stops_weapon.csv", sep=","))

# separate precincts into 
# lt10: less than 10% black
# 1040: 10-40% black
# gt40: greater than 40% black
n.ep.lt10 <- subset(n.ep, Ethnic_Comp_Cat==0, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
n.ep.1040 <- subset(n.ep, Ethnic_Comp_Cat==1, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
n.ep.gt40 <- subset(n.ep, Ethnic_Comp_Cat==2, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
y.ep.lt10 <- subset(y.ep, Ethnic_Comp_Cat==0, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
y.ep.1040 <- subset(y.ep, Ethnic_Comp_Cat==1, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
y.ep.gt40 <- subset(y.ep, Ethnic_Comp_Cat==2, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))

# lt10 hm fit for model 3 of gelman paper
numIters = 1000
nChains = 4
current.n.ep = n.ep.lt10 
current.y.ep = y.ep.lt10
sample.index <- nrow(current.n.ep)
dataList.2 <- list(sample_index=sample.index, n=current.n.ep$Occurrences, y=current.y.ep$Occurrences, sample_race=current.y.ep$Race_Int)
saf.m3 <- stan(file='./stan/sf_model3.stan', 
               data=dataList.2,
               chains = nChains,
               iter = numIters) 

# extract parameters 
post3 <- extract(saf.m3)
alpha_m3 <- colMeans(post3$alpha)
beta_m3 <- colMeans(post3$beta)
mu_m3 <- colMeans(post3$mu)
epsilon_m3 <- colMeans(post3$epsilon)

# generate samples from model 3
lambdas_m3 = c(1:sample.index)
for (i in 1:sample.index) {
  lambdas_m3[i] = (15/12) * current.n.ep$Occurrences[i] * exp(mu_m3 + alpha_m3[current.n.ep$Race_Int[i]] + beta_m3[ceiling(i/3)] + epsilon_m3[i])
}
samples_m3 = rpois(sample.index, lambdas_m3)
current.y.ep$Y_Pred_M3 = samples_m3
alpha_m3


# lt10 hm fit for model 4
saf.m4 <- stan(file='./stan/sf_model4.stan', 
               data=dataList.2, 
               chains = nChains,
               iter = numIters) 

# extract parameters
post4 <- extract(saf.m4)
alpha_m4 <- colMeans(post4$alpha)
gamma <- colMeans(post4$gamma)
beta_m4 <- colMeans(post4$beta)
mu_m4 <- colMeans(post4$mu)
epsilon_m4 <- colMeans(post4$epsilon)

# generate samples from model 4
lambdas_m4 = c(1:sample.index)
for (i in 1:sample.index) {
  lambdas_m4[i] = (15/12) * exp(gamma * log(current.n.ep$Occurrences[i]) + mu_m4 + alpha_m4[current.n.ep$Race_Int[i]] + beta_m4[ceiling(i/3)] + epsilon_m4[i])
}
samples_m4 = rpois(sample.index, lambdas_m4)
current.y.ep$Y_Pred_M4 = samples_m4
alpha_m4


# lt10 hm fit for model 5
dataList.3 <- list(sample_index=sample.index, n=current.n.ep$Occurrences, y=current.y.ep$Occurrences, sample_race=current.y.ep$Race_Int, eth_pop=current.y.ep$Eth_Pop_In_Precinct)
saf.m5 <- stan(file='./stan/sf_model5.stan', 
               data=dataList.3, 
               iter=numIters, 
               chains=nChains) 

# extract parameters
post5 <- extract(saf.m5)
alpha_m5 <- colMeans(post5$alpha)
theta <- colMeans(post5$theta)
beta_m5 <- colMeans(post5$beta)
mu_m5 <- colMeans(post5$mu)
epsilon_m5 <- colMeans(post5$epsilon)

# generate samples from model 5
lambdas_m5 = c(1:sample.index)
for (i in 1:sample.index) {
  lambdas_m5[i] = (15/12) * theta[i] * exp(mu_m5 + alpha_m5[current.n.ep$Race_Int[i]] + beta_m5[ceiling(i/3)] + epsilon_m5[i])
}
samples_m5 = rpois(sample.index, lambdas_m5)
current.y.ep$Y_Pred_M5 = samples_m5
alpha_m5

# calculate log-likelihood 
calc_avg_ll <- function(samples, lambdas) {
  n = length(samples)
  log_sum = sum(log(dpois(samples, lambdas)))
  avg_ll = log_sum / n
}
m3_avg_ll = calc_avg_ll(current.y.ep$Occurrences, lambdas_m3)
m4_avg_ll = calc_avg_ll(current.y.ep$Occurrences, lambdas_m4)
m5_avg_ll = calc_avg_ll(current.y.ep$Occurrences, lambdas_m5)


# combine ethnic populations with predicted stops for wgan
new.nrow = dim(current.y.ep)[1]/3
eth_pops = matrix(n.ep.lt10$Eth_Pop_In_Precinct, nrow=new.nrow, byrow=TRUE)
recorded_stops = matrix(current.y.ep$Occurrences, nrow=new.nrow, byrow=TRUE)
model_3_stops = matrix(current.y.ep$Y_Pred_M3, nrow=new.nrow, byrow=TRUE)
model_4_stops = matrix(current.y.ep$Y_Pred_M4, nrow=new.nrow, byrow=TRUE)
model_5_stops = matrix(current.y.ep$Y_Pred_M5, nrow=new.nrow, byrow=TRUE)
eth_total = 1:nrow(eth_pops)
stops_prev_year = matrix(n.ep.lt10$Occurrences, nrow=new.nrow, byrow=TRUE)
for (i in 1:nrow(eth_pops)) {
  eth_total[i] = eth_pops[i,1] + eth_pops[i,2] + eth_pops[i,3]
}

pops_and_preds_m3 = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               n_ep_w = stops_prev_year[,1],
                               n_ep_b = stops_prev_year[,2],
                               n_ep_q = stops_prev_year[,3],
                               pred_white = model_3_stops[,1],
                               pred_black = model_3_stops[,2],
                               pred_hisp = model_3_stops[,3])
pops_and_preds_m4 = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               n_ep_w = stops_prev_year[,1],
                               n_ep_b = stops_prev_year[,2],
                               n_ep_q = stops_prev_year[,3],
                               pred_white = model_4_stops[,1],
                               pred_black = model_4_stops[,2],
                               pred_hisp = model_4_stops[,3])
pops_and_preds_m5 = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               n_ep_w = stops_prev_year[,1],
                               n_ep_b = stops_prev_year[,2],
                               n_ep_q = stops_prev_year[,3],
                               pred_white = model_5_stops[,1],
                               pred_black = model_5_stops[,2],
                               pred_hisp = model_5_stops[,3])
pops_and_recorded = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               n_ep_w = stops_prev_year[,1],
                               n_ep_b = stops_prev_year[,2],
                               n_ep_q = stops_prev_year[,3],
                               stops_white = recorded_stops[,1],
                               stops_black = recorded_stops[,2],
                               stops_hisp = recorded_stops[,3])
write.csv(pops_and_preds_m3, file='./data/pops_and_preds_m3.csv', row.names=FALSE)
write.csv(pops_and_preds_m4, file='./data/pops_and_preds_m4.csv', row.names=FALSE)
write.csv(pops_and_preds_m5, file='./data/pops_and_preds_m5.csv', row.names=FALSE)
write.csv(pops_and_recorded, file='./data/safData.csv', row.names=FALSE)
    
