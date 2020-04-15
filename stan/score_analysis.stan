  
data {
  int<lower=0> num_real_samples;
  int<lower=0> num_h_models;
  int<lower=0> num_fake_samples;
  int<lower=0> wins[num_real_samples, num_fake_samples, num_h_models];
}
parameters {
  real<lower=0> mu[num_real_samples];
  real<lower=0> gamma[num_real_samples, num_h_models];
  real<lower=0> sigma[2];
}
transformed parameters {
  real theta[num_real_samples, num_h_models]; 
  
  for (i in 1:num_real_samples){
    for (k in 1:num_h_models){
      theta[i,k] = mu[i] + gamma[i,k];
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
    for (k in 1:num_h_models){
      gamma[i,k] ~ normal(0, sigma[2]);
    }
  }
  
  
}
