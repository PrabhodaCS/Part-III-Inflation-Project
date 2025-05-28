

M = 20
\[Mu] = 1
\[Phi] = 3

\[Gamma] = 20
\[Alpha] = -50
\[Beta] = -0.1
k = 8.5

(* Define A(phi), K(phi), and V(phi) just like in your Mathematica file *)

D2 = \[Gamma]^2 - 4*\[Beta]*\[Alpha]                      (*   \[Phi]^2*((\[Sigma]-12*\[Gamma])^2-4*(\[Epsilon]-12*\[Beta])*(\[Nu]-12*\[Alpha])) *)

A[\[CurlyPhi]_] := \[Beta]*\[CurlyPhi]^2 + \[Gamma]*\[Phi]*\[CurlyPhi] + \[Alpha]*\[Phi]^2
A[\[CurlyPhi]]

K[\[CurlyPhi]_] := 
 M^2*( -((36*\[Beta])/A[\[CurlyPhi]]) + (\[Phi]^2*D2*(k - 6))/(
    2*A[\[CurlyPhi]]^2))
K[\[CurlyPhi]]

(* Compute canonical field mapping and save data to CSV *)
tp[χ_] := NIntegrate[Sqrt[K[x]], {x, 0, χ}];
data = Table[{χ, tp[χ]}, {χ, 0, 2, 0.1}]; (* Adjust range & step size *)
Export["C:/Users/Asus/Documents/Cambridge/Project/Inflation Project/Git Repo/Part-III-Inflation-Project/Mathematica/Data/data.csv", data]
