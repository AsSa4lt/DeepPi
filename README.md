sudo apt-get update
sudo apt-get install libgtest-dev

cd /usr/src/googletest
sudo cmake .
sudo make
sudo cp lib/libgtest*.a /usr/lib/
