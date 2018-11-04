# Collage generator

This little tool generates a collage which perserves the aspect ratios of all input images and fits them into the desired output image resolution.
This program is based on the work of Wu et al.[1]. A few adjustments have been made to gain better and more reliable results, but computation takes longer.

## Usage
For help and all options:
```
python collage_generator.py -h
```
Run example:
```
python collage_generator.py ./example --output example_collage.png --width 4096 --height 2048
```

## Example
For source images reference to the ```example``` folder.
![ExampleCollage](collage.png)

[1] [Wu, Zhipeng, and Kiyoharu Aizawa. "Very fast generation of content-preserved photo collage under canvas size constraint." Multimedia Tools and Applications 75.4 (2016): 1813-1841.](https://www.researchgate.net/publication/269455490_Very_fast_generation_of_content-preserved_photo_collage_under_canvas_size_constraint)
