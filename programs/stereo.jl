# Doesn't work
try
    using TiffImages
catch
    import Pkg;
    Pkg.add("TiffImages")
end
# specify the path to your local image file
img_path = "stereo-corridor_l.tiff"
img = TiffImages.load(img_path)