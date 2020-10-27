MPIEXEC = mpiexec -n 4
PYTHON3 = python3



# cubic
# -----
# simulations
# -----------
outputs/cubic-data.h5: scripts/dedalus-cubic.py
	$(PYTHON3) $< --output $@

outputs/cubic-reference.h5: scripts/dedalus-cubic.py
	$(PYTHON3) $< --save_model --output $@

outputs/cubic-deterministic.h5: scripts/run-deterministic-cubic.py
	$(PYTHON3) $< --output $@

outputs/cubic-posterior-ekf.h5: outputs/cubic-data.h5
	$(PYTHON3) scripts/run-condapprox-cubic.py \
		--data outputs/cubic-data.h5 \
		--output $@

outputs/cubic-posterior-enkf.h5: outputs/cubic-data.h5
	$(MPIEXEC) $(PYTHON3) scripts/run-condmc-cubic.py \
		--data outputs/cubic-data.h5 \
		--output $@

# plots
# -----
figures/cubic-prior-consistency.pdf:
	$(PYTHON3) scripts/plot-cubic-prior-consistency.py --output $@

figures/cubic-posterior.pdf:
	$(PYTHON3) scripts/plot-cubic-posterior.py --output $@

figures/cubic-heatmap.png:
	$(PYTHON3) scripts/plot-cubic-heatmap.py --output $@

figures/cubic-parameters.pdf:
	$(PYTHON3) scripts/plot-cubic-parameters.py \
		--simulation outputs/cubic-posterior-enkf.h5 \
		--output $@





# tank
# ----
# simulations
outputs/tank-deterministic.h5:
	$(PYTHON3) scripts/run-deterministic-tank.py --output $@

outputs/tank-posterior-enkf.h5:
	$(MPIEXEC) $(PYTHON3) scripts/run-condmc-tank.py --output $@

outputs/tank-posterior-ekf.h5:
	$(PYTHON3) scripts/run-condapprox-tank.py --output $@

# plots
figures/tank-posterior.pdf: outputs/tank-posterior-enkf.h5
	$(PYTHON3) scripts/plot-expdata-posterior.py \
		--simulation outputs/tank-posterior-enkf.h5 \
		--output $@

figures/tank-heatmap.pdf:
	$(PYTHON3) scripts/plot-expdata-heatmap.py \
		--simulation outputs/tank-posterior-enkf.h5 \
		--output $@

figures/tank-parameters.pdf:
	$(PYTHON3) scripts/plot-expdata-parameters.py \
		--simulation outputs/tank-posterior-enkf.h5 \
		--output $@




# fenics examples
# ---------------
outputs/ks-posterior.h5:
	$(PYTHON3) scripts/ks.py --output $@

outputs/burgers-dgp.h5:
	$(PYTHON3) scripts/burgers_2d.py --output $@

outputs/burgers-posterior.h5:
	$(PYTHON3) scripts/burgers_2d_ekf.py --output $@

outputs/burgers-1d-posterior.h5:
	$(PYTHON3) scripts/burgers_1d.py --output $@

# dummy target to make all plots in a single script
ks_plots:
	$(PYTHON3) scripts/plot_ks.py

burgers_2d_plots:
	$(PYTHON3) scripts/plot_burgers_2d.py

burgers_1d_plots:
	$(PYTHON3) scripts/plot_burgers_1d.py
