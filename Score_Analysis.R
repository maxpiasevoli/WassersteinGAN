library(rstan)
library(ggplot2)
library(reshape2)
library(rjson)

setwd("C:/Users/maxpi/Documents/SML310/WassersteinGAN")

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
behavioral.analysis <- stan(file='./stan/score_analysis2.stan', data=dataList, iter=1000, chains=4) # model with intercept
post <- extract(behavioral.analysis)
mu <- colMeans(post$mu)
rho <- colMeans(post$rho)
sigma <- colMeans(post$sigma)

# save means
write.csv(rho, file='./data/behavioral_rho.csv', row.names=FALSE)
write.csv(mu, file='./data/behavioral_mu.csv', row.names=FALSE)
write.csv(sigma, file='./data/behavioral_sigma.csv', row.names=FALSE)
# save all estimates 
write.csv(post$rho, file='./data/behavioral_all_rho.csv', row.names=FALSE)
write.csv(post$mu, file='./data/behavioral_all_mu.csv', row.names=FALSE)
write.csv(post$sigma, file='./data/behavioral_all_sigma.csv', row.names=FALSE)

samples <- data.frame(Logit = post$rho[,1], Log = post$rho[,2], LOTM = post$rho[,3], 
                      LFSM = post$rho[,4], NTLOTM = post$rho[,5])

library(tidyverse)
samples.long <- samples %>% pivot_longer(cols = c(Logit, Log, LOTM, LFSM, NTLOTM), 
                                         values_to = "rho", names_to = "Model")
ggplot(samples.long) +
  geom_histogram(mapping = aes(x = rho, fill = Model),
                 alpha = 0.5, position = "identity", binwidth=0.01) +
  xlab("Estimated Rho Value") +
  ggtitle("Estimated Rho Values by Model") +
  ylab("Count")

ggsave("./experiment_images/rho_by_model.png")
