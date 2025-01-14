rm -rf parlastress
git clone https://github.com/clarinsi/parlastress.git
mkdir -p data/input
cp -r parlastress/prim_stress/* data/input/
rm -rf parlastress