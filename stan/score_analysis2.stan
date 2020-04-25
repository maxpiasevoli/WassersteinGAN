  
data {
  int<lower=0> num_real_samples;
  int<lower=0> num_h_models;
  int<lower=0> num_fake_samples;
  int<lower=0> wins[num_real_samples, num_fake_samples, num_h_models];
}
parameters {
  real mu[num_real_samples];
  real gamma[num_h_models];
  real<lower=0> sigma[2];
}
transformed parameters {
  real theta[num_real_samples, num_h_models]; 
  
  for (i in 1:num_real_samples){
    for (k in 1:num_h_models){
      theta[i,k] = exp(mu[i] + gamma[k]);
    }
  }
  
}
model {
  for (i in 1:num_real_samples){
    for (j in 1:num_fake_samples){
      for (k in 1:num_h_models){
        wins[i,j,k] ~ poisson(theta[i,k]);
      }
    }
  }
  
  for (i in 1:num_real_samples){
    mu[i] ~ normal(0, sigma[1]);
  }
  for (k in 1:num_h_models){
    gamma[k] ~ normal(0, sigma[2]);
  }
  
  
}
