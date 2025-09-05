# Overview

This is a working version of the ARAP 2D Shape manipulation in Python and PyQt5. 
To change the `.obj` file, press `f` and the choose the `.obj` file. 

## Usage

* Right-click on the vertices of mesh. A red dot will appear on that part. Create at least 2 of these.
* Then, hold left button on one of the red dots and begin moving the mesh.

## Notes

* `app.py` - uses scipy for LU factorization
* `app_old.py` - does not use scipy for LU factorization


## TODO

* Refactor to make it maintenable
* Add controls just like tsugite
* Possible convert to Kotlin for a drawing app - animation app?