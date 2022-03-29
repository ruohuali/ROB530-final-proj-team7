# General Guideline
This is the folder demonstrating the basic usage of socket in communication

# Code Explanation
This folder contains a simple client and a simple server (in .py and .cpp )
Basically, the client will send an array representing a pseduo image to the
server, and the server will send back a random array

# Usage
To check the demonstration on your local PC, do the following:
1. create a new folder called "build"
2. go into "build", run "cmake ../src"
3. (Python server) copy "server.py" from "src" into "build"
4. (Python server) open a terminal, run "server.py", using, for example, "python3" in Linux
5. open **another** terminal, run "make"
6. in the same terminal, run "./client", then you should see the outputs from both the client and the server
