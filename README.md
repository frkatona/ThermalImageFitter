# Thermal Image Plotting and Fitting

## todo

- [x] fix line centering
  - [x] output the image with the line overlaid to see if it's stable and within the image bounds (uncomment the `TroubleshootLinePosition` function)
  - [x] make the position consistent (instead of using max: (1) find which pixel has the highest gaussian kernel weight or (2) pick the high point of image ~3 and use that pixel location going forward) --> ended up (1) zeroing outside of the region-of-interest bounds and then finding the line which summed to the highest value
  - [x] increase line length by enough to capture T_infinity
- [ ] remove smoothing and instead try Ben's Gaussian hybrid fit method (linear + sharp inner gaussian + broad outer gaussian)
  - [ ] replace the over-exposed max T scatterplot values with the amplitude of the fit trace
  - [ ] test Ben's method on a proper thermal profile by artificially saturating it and seeing if the fit recovers the original profile
- [ ] make a separate figure for the "cooling" profiles (starting from the final "heating" profile)
- [ ] polish formatting (font/size, titles, grids, labels, positions, point size, fit equations)
- [ ] get into LaTeX, maybe try plotly

## careful when using script

- be sure to check the min and max temperatures are properly recorded from the images and they are consistent across images (otherwise the script needs refactored)

## info

- thermal images are 640 x 480 pixels
- x position increases left to right and y position increases top to bottom and x and y sometimes seem unintuitively swapped in array references