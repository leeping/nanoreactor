#!/bin/bash

cat <<EOF > plot-rc.gp

set encoding iso_8859_1
set terminal svg enhanced size 250,200 fsize 10
set output "irc_energy.svg"
unset xtics
unset ytics
set xlabel 'Reaction Coordinate (\305)' offset 0, 0.8
set ylabel 'Energy (kcal/mol)' offset 1, 0
xmin = 0
xmax = `grep -v "#" plot.nrg | awk '{printf "%.2f\n", $1}' | sort -n | tail -1 `
ymax = `grep -v "#" plot.nrg | awk '{printf "%.2f\n", $2}' | sort -n | tail -1 `
ymaxi = `grep -v "#" plot.nrg | awk '{printf "%i\n", $2}' | sort -n | tail -1 `
ymin = `grep -v "#" plot.nrg | awk '{printf "%.2f\n", $2}' | sort -n | head -1 `
yrxn = `tail -1 plot.nrg | awk '{printf "%i\n", $2}' `
rx = `grep -v "#" plot.nrg | head -1 | awk '{print $1}'`
px = `grep -v "#" plot.nrg | tail -1 | awk '{print $1}'`
tx = `grep -v "#" plot.nrg | sort -nk 2 | tail -1 | awk '{print $1}'`
ry = `grep -v "#" plot.nrg | head -1 | awk '{print $2}'`
py = `grep -v "#" plot.nrg | tail -1 | awk '{print $2}'`
ty = `grep -v "#" plot.nrg | sort -nk 2 | tail -1 | awk '{print $2}'`
dx = xmax - xmin
dy = ymax - ymin

set xrange[xmin-dx*0.1-0.1:xmax+dx*0.1+0.1]
set yrange[ymin-dy*0.1-0.1:ymax+dy*0.1+0.1]

set xtics (xmin, xmax)
set ytics (yrxn, ymaxi)

set label at rx, ry "" point pt 6 ps 1.4
set label at tx, ty "" point pt 12 ps 1.4
set label at px, py "" point pt 6 ps 1.4
p "plot.nrg" notitle w l lc 1 lt 1 lw 3

EOF

gnuplot plot-rc.gp
