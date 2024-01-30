import jax.numpy as np

def scatterer_positions(amp_gaus, sig_gaus, spacing_scat, num_scat, dist_mirror):
    num_scat_per_row = np.floor(num_scat/2)
    additional_left = num_scat-num_scat_per_row*2

    positions_y = np.arange(num_scat_per_row, dtype=float) * spacing_scat
    positions_y-= np.mean(positions_y)
    positions_x = 0.5*dist_mirror + gaussian(positions_y, amp_gaus, sig_gaus)

    positions_y = np.concatenate([positions_y, positions_y])
    positions_x = np.concatenate([positions_x,-positions_x])
    if additional_left:
        positions_y = np.append(positions_y, [0])
        positions_x = np.append(positions_x, [0])
    return positions_x, positions_y, np.zeros_like(positions_x)

def gaussian(y, amp_gaus, sig_gaus):
    return amp_gaus*np.exp(-y**2/(2*sig_gaus**2))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.scatter(*scatterer_positions(100, 300, 90, 16, 100)[0:2])
    plt.show()