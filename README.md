# Overview

This is a working version of the ARAP 2D Shape manipulation in Python and PyQt5. Original paper is [here](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/rigid.pdf)


## Usage

* Run `$ python main.py` 
* To change the `.obj` file, press `f` and the choose the `.obj` file. 
* Right-click on the vertices of mesh. A red dot will appear on that part. Create at least 2 of these.
* Then, hold left button on one of the red dots and begin moving the mesh.


## Notes

* An intuitive intro to ARAP is [here](https://erkaman.github.io/posts/sorkine2007.html)
* `app.py` - uses scipy for LU factorization
* `misc/app_old.py` - does not use scipy for LU factorization

## How to read a .obj file

* `v` - vertices of the triangles 
* `vn` - vertex normals of the triangles
* `vt` - vertex textures
* `f` - faces. the format is like this `{vertex index}/{vertex texture coordinate index}/{vertex normal index}`.


## TODO

* Refactor to make it maintenable
* Add controls just like tsugite
* Have it two modes - drawing and picture upload
* Possible convert to Kotlin for a drawing app - animation app?