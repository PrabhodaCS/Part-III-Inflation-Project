import subprocess
import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Define parameters
M = 20
mu = 1
phi0 = 3
gamma = 20
alpha = -50
beta = -0.1
k = 8.5

End = 2
stepsize = 0.1

# Create a new Mathematica script with updated parameters
mathematica_script = f"""

M = {M}
\[Mu] = {mu}
\[Phi] = {phi0}

\[Gamma] = {gamma}
\[Alpha] = {alpha}
\[Beta] = {beta}
k = {k}

(* Define A(phi), K(phi), and V(phi) just like in your Mathematica file *)

D2 = \[Gamma]^2 - 4*\[Beta]*\[Alpha]                      (*   \
\[Phi]^2*((\[Sigma]-12*\[Gamma])^2-4*(\[Epsilon]-12*\[Beta])*(\[Nu]-\
12*\[Alpha])) *)

A[\[CurlyPhi]_] := \[Beta]*\[CurlyPhi]^2 + \
\[Gamma]*\[Phi]*\[CurlyPhi] + \[Alpha]*\[Phi]^2
A[\[CurlyPhi]]

K[\[CurlyPhi]_] := 
 M^2*( -((36*\[Beta])/A[\[CurlyPhi]]) + (\[Phi]^2*D2*(k - 6))/(
    2*A[\[CurlyPhi]]^2))
K[\[CurlyPhi]]

(* Compute canonical field mapping and save data to CSV *)
tp[χ_] := NIntegrate[Sqrt[K[x]], {{x, 0, χ}}];
data = Table[{{χ, tp[χ]}}, {{χ, 0, {End}, {stepsize}}}]; (* Adjust range & step size *)
Export["C:/Users/Asus/Documents/Cambridge/Project/Inflation Project/Git Repo/Part-III-Inflation-Project/Mathematica/Data/data.csv", data]
"""

# Save Mathematica script to a file
mathematica_filename = "C:/Users/Asus/Documents/Cambridge/Project/Inflation Project/Git Repo/Part-III-Inflation-Project/Mathematica/RunScript.wl"
with open(mathematica_filename, "w", encoding="utf-8") as f:
    f.write(mathematica_script)

# Run Mathematica script using subprocess
subprocess.run([r"C:\Program Files\Wolfram Research\Wolfram\14.2\WolframNB.exe", "-script", mathematica_filename])

# Read output CSV
data_file = r"C:\Users\Asus\Documents\Cambridge\Project\Inflation Project\Git Repo\Part-III-Inflation-Project\Mathematica\Data\data.csv"

varphi = []
chi = []

with open(data_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        varphi.append(float(row[0]))
        chi.append(float(row[1]))

#ok so, now i have a list of values of \varphi and \tilde{\varphi}. now i should be able to get the values of V(phi) as a function of \tilde{\varphi} and then plot it in python ? ig.
varphi = np.array(varphi)
chi = np.array(chi)


def V(x):
    A = beta*x**2 + gamma*phi0*x + alpha*phi0**2
    return 0.5 * (mu*phi0)**2 * (x**2)/A**2



# Plot the results
plt.plot(varphi, chi)
plt.xlabel(r"$\varphi$")
plt.ylabel(r"$\chi$")
plt.title(r"$\chi$ vs $\varphi$")
plt.grid()
plt.show()


plt.plot(varphi, V(varphi))
plt.xlabel(r"$\varphi$")
plt.ylabel(r"$V(\varphi)$")
plt.title(r"V($\varphi$) vs $\varphi$")
plt.grid()
plt.show()

plt.plot(varphi, V(chi))
plt.xlabel(r"$\chi$") 
plt.ylabel(r"$V(\chi)$")
plt.title(r"$V(\chi)$ vs $\chi$")   
plt.grid()
plt.show()
