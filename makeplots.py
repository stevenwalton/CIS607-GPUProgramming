import matplotlib.pyplot as plt

x = [0.1, 0.05, 0.025, 0.0125]
error = [0.055232, 0.024741, 0.011982, 0.005942]
ratio = [2.232350, 2.064825, 2.016699, 2.004207]
rate = [-4.178364, -5.336927, -6.382947, -7.394943]

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] =  30
plt.rcParams['axes.labelsize'] = 20 
plt.rcParams['axes.titlesize'] = 26

fig, (ax0, ax1, ax2) = plt.subplots(figsize=(18, 16), tight_layout=True,
                                    nrows=3, ncols=1, sharex=True)

ax0.invert_xaxis()

plt.suptitle("CPU Code")
#ax0.set_title(r'$error_{\Delta x}$')
ax0.set_xlabel(r'$\Delta x$')
ax0.set_ylabel("error")
ax0.set_xticks(x)
ax0.semilogx(x, error)

ax1.set_title(r'ratio: $error_{\Delta x}/ error_{\Delta x / 2}$')
#ax1.set_xlabel(r'$\Delta x$')
ax1.set_ylabel("ratio")
ax1.plot(x, ratio)

ax2.set_title(r'rate: $\log_2(ratio)$')
ax2.set_xlabel(r'$\Delta x$')
ax2.set_ylabel("rate")
ax2.plot(x, rate)

plt.savefig("CPU.png")

error = [0.390671, 0.492058, 0.583328, 0.64081]
ratio = [0.793954, 0.843534, 0.910197, 0.952675]
rate = [-1.355974, -1.023101, -0.777620, -0.641871]

fig, (ax0, ax1, ax2) = plt.subplots(figsize=(18, 16), tight_layout=True,
                                    nrows=3, ncols=1, sharex=True)

ax0.invert_xaxis()

plt.suptitle("GPU Code")
#ax0.set_title(r'$error_{\Delta x}$')
ax0.set_xlabel(r'$\Delta x$')
ax0.set_ylabel("error")
ax0.set_xticks(x)
ax0.semilogx(x, error)

ax1.set_title(r'ratio: $error_{\Delta x}/ error_{\Delta x / 2}$')
#ax1.set_xlabel(r'$\Delta x$')
ax1.set_ylabel("ratio")
ax1.plot(x, ratio)

ax2.set_title(r'rate: $\log_2(ratio)$')
ax2.set_xlabel(r'$\Delta x$')
ax2.set_ylabel("rate")
ax2.plot(x, rate)

plt.savefig("GPU.png")
