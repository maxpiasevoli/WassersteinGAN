  
data {
  int<lower=0> n_dogs;
  int<lower=0> n_trials;
  int<lower=0,upper=1> y[n_dogs,n_trials];
}
parameters {
  vector[1] alpha;
  vector[1] gamma;
}
transformed parameters {
  matrix[n_dogs,n_trials] p;
  
  for (j in 1:n_dogs) {
    for (t in 1:n_trials) {
      p[j,t] = alpha[1]/t + gamma[1];
    }
  }
}
model {
  for (i in 1:n_dogs) {
    for (j in 1:n_trials)
      y[i,j] ~ bernoulli_logit(p[i,j]);
  }
}