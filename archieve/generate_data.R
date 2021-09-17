library("IDE")

# simulate from a IDE model
SIM1 <- simIDE(T=10, nobs=100, k_spat_invariant=1)
SIM2 <- simIDE(T=10, nobs=100, k_spat_invariant=1)

print(SIM1$g_truth)
print(SIM2$g_truth)

IDEmodel <- IDE(f=z~s1+s2, data=SIM1$z_STIDF, dt=as.difftime(1, units='days'), grid_size=41)
