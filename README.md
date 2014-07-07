#KernelSanders

![the_kernel](http://25.media.tumblr.com/tumblr_lab6dibWmY1qa95zho1_500.gif)

A CUDA raytracer and raymarcher with DirectX 11 interoperability.
KernelSanders is able to render textured meshes, geometric primitives and 3d fractals in
real time. Meshes and primitives are lit from several light sources and can cast shadows and have reflective surfaces.
When a mesh is loaded for the first time the application will construct a kD-tree for
it, using the SAH-algorithm. This tree is then cached on disk for future uses.

An infinite starry sky is rendered using raymarched Mandelbulb fractals. Each fractal
is unique (based on placement) and can be reached by navigating using the keyboard and mouse or an Xbox360 gamepad.

Video (raymarcher part): https://www.youtube.com/watch?v=dHV4Lj_LJp0 

![mandel1](https://dl.dropboxusercontent.com/u/2014021/raytracer/raytracer-report/img/mandel1.png)

![mandel2](https://dl.dropboxusercontent.com/u/2014021/raytracer/raytracer-report/img/mandel2.png)

![mandel3](https://dl.dropboxusercontent.com/u/2014021/raytracer/raytracer-report/img/mandel3.png)

![scene](https://dl.dropboxusercontent.com/u/2014021/raytracer/raytracer-report/img/scene.png)

![frac](https://dl.dropboxusercontent.com/u/2014021/raytracer/raytracer-report/img/frac4.png)

![kd](https://dl.dropboxusercontent.com/u/2014021/raytracer/raytracer-report/img/hi_kd.png)

![stars](https://dl.dropboxusercontent.com/u/2014021/raytracer/raytracer-report/img/starfield.png)


##External dependencies
- Assimp
- CUDA
- FreeImage
- GLM
- OIS
- Visual Leak Detector

##License
- KernelSanders license is found in licenseKernelSanders.md.
- Parts of the engine(mainly the libs Graphics, Util and Context) are also based on the Amalgamation engine of which I'm a co-author, for those, see licenseOther.md.


