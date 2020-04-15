library(rstan)
library(ggplot2)
library(reshape2)

setwd("C:/Users/maxpi/Documents/SML310/WassersteinGAN")

# NOT CORRECT 
wins_json <- fromJSON(file='./data/behavioral_wins.json')
wins <- array(as.numeric(unlist(wins_json)), dim=c(5, 1000, 10))
wins[,1,1]

wins <- aperm(wins, c(3,2,1))

wins[1,1:20,1]
wins[2,1:20,2]
wins[1:10,1,1]

num.real.samples = 10
num.h.models = 5
num.fake.samples = 1000
dataList <- list(num_real_samples=num.real.samples, num_h_models=num.h.models, num_fake_samples=num.fake.samples, wins=wins)
behavioral.analysis <- stan(file='./stan/score_analysis.stan', data=dataList, iter=1000, chains=4) # model with intercept
print(behavioral.analysis, pars = c("beta","lp__"))
post <- extract(behavioral.analysis)
mu <- colMeans(post$mu)
gamma <- colMeans(post$gamma)

write.csv(gamma, file='./data/behavioral_gamma.csv', row.names=FALSE)
write.csv(mu, file='./data/behavioral_mu.csv', row.names=FALSE)