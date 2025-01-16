function U_SlantedGauss(x::Vector)
  return (sum(x.^2) - x[1]*x[2]) / 2
end

function U_banana(x::Vector)
  mean_x2 = (x[1]^2 - 1)
  return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
end

function funnel(d::Int=10, σ::Float64=3.0, clip_y::Int=11)
  function neg_energy(x::Vector)
      v = x[1]
      log_density_v = logpdf(Normal(0.0, 3.0), v)
      variance_other = exp(v)
      other_dim = d - 1
      cov_other = I * variance_other
      mean_other = zeros(other_dim)
      log_density_other = logpdf(MvNormal(mean_other, cov_other), x[2:end])
      return - log_density_v - log_density_other
  end

  function sample_data(n_samples::Int)
      # sample from Nd funnel distribution
      y = clamp.(σ * randn(n_samples, 1), -clip_y, clip_y)
      x = randn(n_samples, d - 1) .* exp.(-y / 2)
      return hcat(.- y, x)
  end

  return neg_energy, sample_data
end