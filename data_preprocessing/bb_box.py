path_to_image = "/scratch/p302386/dataLongitudinal/UMCG-0020715/"

def get_bounding_box(mask):
    # Get the bounding box from the mask
    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(mask, mask)
    bounding_box = label_stats.GetBoundingBox(1)  # Bounding box for label value 1 (assuming binary mask)
    
    return bounding_box

# Load the mask and image
mask_path = "mask.nii.gz"
image_path = "image.nii.gz"

mask = sitk.ReadImage(mask_path)
image = sitk.ReadImage(image_path)

# Get the bounding box
bounding_box = get_bounding_box(mask)

# Extract the bounding box region from the image
cropped_image = sitk.RegionOfInterest(image, bounding_box)