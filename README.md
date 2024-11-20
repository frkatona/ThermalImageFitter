# todo
- [ ] fix line centering
  - [ ] output the image with the line overlaid to see if it's going offscreen
  - [ ] make the position consistent (instead of using max: (1) find which pixel has the highest gaussian kernel weight or (2) pick the high point of image ~3 and use that pixel location going forward)
    - create the line but scan across all the positions within a certain bounding box and use the center position corresponding to the line with the greatest summed intensity across its length (TL: 300, 226, BL: 311, 297, 380, 226, 380, 297)
  - [ ] increase line length by ~20%
- [ ] remove smoothing and instead try Ben's Gaussian hybrid fit method (linear + sharp inner gaussian + broad outer gaussian)
- [ ] make a separate figure for the "cooling" profiles (starting from the final "heating" profile)
- [ ] polish formatting (font/size, titles, grids, labels, positions, point size, fit equations)
- [ ] get into LaTeX, maybe try plotly

# careful when using script
- be sure to check the min and max temperatures are properly recorded from the images and they are consistent across images (otherwise the script needs refactored)

# info
- thermal images are 640 x 480 pixels
- x position increases left to right and y position increases top to bottom and x and y sometimes seem unintuitively swapped in array references