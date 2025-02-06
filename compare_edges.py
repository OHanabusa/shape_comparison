import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def load_and_preprocess_image(image_path, crop_ratio=None):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduced blur kernel
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Use Canny edge detection with adjusted thresholds
    edges = cv2.Canny(enhanced, 30, 150)  # Lower threshold for better edge detection
    
    # Dilate edges to connect any gaps
    kernel = np.ones((2,2), np.uint8)  # Smaller kernel
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours with different method
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Get the largest contour (main shape)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Draw both the edge detection and contour for debugging
    debug_img = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, [largest_contour], -1, (0, 255, 0), 2)
    
    # Create edge-only version
    edge_only = create_edge_only_image(largest_contour, debug_img.shape[:2])
    
    # Save debug image with original name
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if crop_ratio is not None:
        # Crop the debug image directly
        height = debug_img.shape[0]
        start_y = int(height * crop_ratio[0])
        end_y = int(height * crop_ratio[1])
        debug_img_cropped = debug_img[start_y:end_y, :]
        edge_only_cropped = edge_only[start_y:end_y, :]
        
        debug_name = f"debug_{base_name}_cropped.png"
        edge_only_name = f"edge_only_{base_name}_cropped.png"
        
        cv2.imwrite(debug_name, debug_img_cropped)
        cv2.imwrite(edge_only_name, edge_only_cropped)
    else:
        debug_name = f"debug_{base_name}.png"
        edge_only_name = f"edge_only_{base_name}.png"
        
        cv2.imwrite(debug_name, debug_img)
        cv2.imwrite(edge_only_name, edge_only)
    
    # Crop the contour if ratio is provided
    if crop_ratio is not None:
        height = img.shape[0]
        start_y = int(height * crop_ratio[0])
        end_y = int(height * crop_ratio[1])
        # Adjust contour points
        largest_contour = largest_contour - [0, start_y]
        mask = (largest_contour[:,:,1] >= 0) & (largest_contour[:,:,1] <= (end_y - start_y))
        largest_contour = largest_contour[mask]
    
    return edge_only_name

def create_edge_only_image(contour, size):
    # Create a black background
    edge_only = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    # Draw the contour in green
    cv2.drawContours(edge_only, [contour], -1, (0, 255, 0), 2)
    return edge_only

def center_and_scale_contour(img):
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, None
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Get center of the image
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    
    # Calculate translation
    tx = center_x - (x + w//2)
    ty = center_y - (y + h//2)
    
    # Create translation matrix
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    centered = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # Update contour position
    contour += np.array([tx, ty])
    
    return centered, contour

# def calculate_similarity(template, target):
#     # Make sure both images have the same size
#     target_size = (400, 400)  # Fixed size for comparison
#     img1_resized = cv2.resize(template, target_size)
#     img2_resized = cv2.resize(target, target_size)
    
#     # Center both images based on their contours
#     img1_centered, contour1 = center_and_scale_contour(img1_resized)
#     img2_centered, contour2 = center_and_scale_contour(img2_resized)
    
#     if contour1 is None or contour2 is None:
#         return 0, None, None, None
    
#     # Convert to grayscale for SSIM
#     gray1 = cv2.cvtColor(img1_centered, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2_centered, cv2.COLOR_BGR2GRAY)
    
#     # Calculate similarity
#     similarity, _, _ = calculate_similarity(gray1, gray2)
    
#     # Create overlay image for visualization
#     overlay = np.zeros_like(img1_centered)
#     cv2.drawContours(overlay, [contour1], -1, (0, 255, 0), 2)  # First contour in green
#     cv2.drawContours(overlay, [contour2], -1, (0, 0, 255), 2)  # Second contour in red
    
#     return similarity, img1_centered, img2_centered, overlay

def calculate_similarity(template_gray, target_gray):
    # Ensure images are grayscale
    if len(template_gray.shape) > 2:
        template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)
    if len(target_gray.shape) > 2:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)

    # Normalize images to full range
    template_gray = cv2.normalize(template_gray, None, 0, 255, cv2.NORM_MINMAX)
    target_gray = cv2.normalize(target_gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply adaptive thresholding
    template_bin = cv2.adaptiveThreshold(template_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    target_bin = cv2.adaptiveThreshold(target_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

    # Calculate SSIM similarity
    ssim_score = ssim(template_gray, target_gray, data_range=255)
    ssim_similarity = min(100, max(0, (ssim_score + 1) * 50))  # Convert from [-1,1] to [0,100]

    # Calculate area similarity
    template_area = cv2.countNonZero(template_bin)
    target_area = cv2.countNonZero(target_bin)
    area_ratio = min(template_area, target_area) / max(template_area, target_area)
    area_similarity = min(100, max(0, area_ratio * 100))

    # Calculate perimeter similarity using edge detection
    template_edges = cv2.Canny(template_gray, 50, 150)
    target_edges = cv2.Canny(target_gray, 50, 150)
    template_perimeter = cv2.countNonZero(template_edges)
    target_perimeter = cv2.countNonZero(target_edges)
    perimeter_ratio = min(template_perimeter, target_perimeter) / max(template_perimeter, target_perimeter)
    perimeter_similarity = min(100, max(0, perimeter_ratio * 100))

    # Calculate overlap similarity
    kernel = np.ones((3,3), np.uint8)
    template_dilated = cv2.dilate(template_bin, kernel, iterations=1)
    target_dilated = cv2.dilate(target_bin, kernel, iterations=1)
    
    overlap = cv2.bitwise_and(template_dilated, target_dilated)
    union = cv2.bitwise_or(template_bin, target_bin)
    
    if cv2.countNonZero(union) > 0:
        overlap_similarity = min(100, max(0, 100 * cv2.countNonZero(overlap) / cv2.countNonZero(union)))
    else:
        overlap_similarity = 0

    # Combine all similarity metrics with adjusted weights
    #類似度の測定値の比重設定
    similarity = min(100, (
        area_similarity * 0.1 +      # Increased weight for area
        perimeter_similarity * 0.1 +  # Moderate weight for perimeter
        ssim_similarity * 0.5 +       # Moderate weight for SSIM
        overlap_similarity * 0.3      # Increased weight for overlap
    ))

    # Print individual scores for debugging
    # print(f"Area similarity: {area_similarity:.2f}")
    # print(f"Perimeter similarity: {perimeter_similarity:.2f}")
    # print(f"SSIM similarity: {ssim_similarity:.2f}")
    # print(f"Overlap similarity: {overlap_similarity:.2f}")
    # print(f"Final similarity: {similarity:.2f}")

    return similarity

def calculate_similarity_at_position(template, target, x, y, scale):
    """Calculate similarity between template and a section of target at given position and scale"""
    # Scale template while maintaining aspect ratio
    template_height, template_width = template.shape[:2]
    scaled_size = (int(template_width * scale), int(template_height * scale))
    template_scaled = cv2.resize(template, scaled_size)
    
    # Extract region from target
    region_height, region_width = template_scaled.shape[:2]
    if y + region_height > target.shape[0] or x + region_width > target.shape[1]:
        return 0, None, None
    
    region = target[y:y+region_height, x:x+region_width]
    
    # Convert both to grayscale
    if len(template_scaled.shape) == 3:
        template_gray = cv2.cvtColor(template_scaled, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_scaled
        
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        region_gray = region
    
    # Calculate similarity
    similarity = calculate_similarity(template_gray, region_gray)    
    return similarity, template_scaled, region

def create_comparison_image(template, target, best_template, best_match, position, scale, similarity):
    padding = 20
    
    # Get sizes for each image while maintaining aspect ratio
    max_section_width = 400
    max_section_height = 400

    template_resized = resize_maintain_aspect(template, max_section_width, max_section_height)
    target_resized = resize_maintain_aspect(target, max_section_width, max_section_height)
    best_template_resized = resize_maintain_aspect(best_template, max_section_width, max_section_height)
    best_match_resized = resize_maintain_aspect(best_match, max_section_width, max_section_height)

    # Calculate total size needed
    max_height = max(template_resized.shape[0], target_resized.shape[0], 
                    best_template_resized.shape[0], best_match_resized.shape[0])
    total_width = max(template_resized.shape[1], target_resized.shape[1]) * 3 + padding * 4
    total_height = max_height + padding * 2

    # Create the comparison image
    comparison = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # Helper function to center image in its section
    def place_image_centered(canvas, img, x, y, section_width, section_height):
        h, w = img.shape[:2]
        start_x = x + (section_width - w) // 2
        start_y = y + (section_height - h) // 2
        canvas[start_y:start_y+h, start_x:start_x+w] = img

    # Calculate section width for centering
    section_width = (total_width - padding * 4) // 3

    # Place original images in first row
    y1 = padding
    place_image_centered(comparison, template_resized, padding, y1, section_width, max_height)
    place_image_centered(comparison, target_resized, padding * 2 + section_width, y1, section_width, max_height)

    # Create and place the overlay visualization
    combined = target.copy()
    mask = np.zeros_like(target)
    template_height, template_width = best_template.shape[:2]
    
    # Create mask for template
    template_region = mask[position[1]:position[1]+template_height, 
                         position[0]:position[0]+template_width]
    template_region[:] = best_template

    # Create overlay
    mask_binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0
    combined[mask_binary] = [0, 0, 255]  # Set matching area to blue
    
    # Add template outline
    template_contours, _ = cv2.findContours(cv2.cvtColor(best_template, cv2.COLOR_BGR2GRAY), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
    
    if template_contours:
        # Offset contours to match position
        offset_contours = [cnt + [position[0], position[1]] for cnt in template_contours]
        cv2.drawContours(combined, offset_contours, -1, (255, 0, 0), 2)  # Draw template outline in red

    # Draw rectangle around matching region
    cv2.rectangle(combined, 
                 (position[0], position[1]), 
                 (position[0] + template_width, position[1] + template_height), 
                 (0, 255, 0), 3)

    combined_resized = resize_maintain_aspect(combined, max_section_width, max_section_height)

    # Place matched regions and overlay in second row
    # y2 = y1 + max_height + padding
    # place_image_centered(comparison, best_template_resized, padding, y2, section_width, max_height)
    # place_image_centered(comparison, best_match_resized, padding * 2 + section_width, y2, section_width, max_height)
    place_image_centered(comparison, combined_resized, padding * 3 + section_width * 2, y1, section_width, max_height)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "experiment", (padding , padding-5), font, 0.5, (0, 255, 0), 1)
    cv2.putText(comparison, "simulation", (padding * 2 + section_width , padding-5), font, 0.5, (0, 255, 0), 1)
    cv2.putText(comparison, f"Similarity: {similarity:.2f}%", (padding*3 + section_width * 2, y1-5), font, 0.4, (0, 255, 0), 1)
    # cv2.putText(comparison, f"Scale: {scale:.2f}, Pos: {position}", (padding * 2 + section_width, y2-5), font, 0.7, (0, 255, 0), 2)

    return comparison

def find_best_match(template, target, scale_range=(0.5, 1.0, 0.1), step_size=20):
    """Find the best matching position and scale of template within target"""
    best_similarity = 0
    best_match = None
    best_template = None
    best_position = None
    best_scale = None
    
    target_height, target_width = target.shape[:2]
    template_height, template_width = template.shape[:2]
    
    # Calculate center position
    center_y = target_height // 2 - template_height // 2
    center_x = target_width // 2 - template_width // 2
    
    print(f"Target center: ({center_x}, {center_y})")
    
    # Calculate search range (±20% from center)
    search_range_y = int(target_height * 0.2)
    search_range_x = int(target_width * 0.3)
    
    print(f"Search range: ±{search_range_x} pixels horizontally, ±{search_range_y} pixels vertically")
    
    for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]):
        print(f"Checking scale: {scale}")
        # Scale template
        scaled_size = (int(template_width * scale), int(template_height * scale))
        
        # Skip if scaled template is larger than target
        if scaled_size[0] > target_width or scaled_size[1] > target_height:
            continue
        
        # Calculate search boundaries around center
        #捜索範囲設定
        start_y = 170
        end_y = 180
        start_x = 40
        end_x = 50
        # start_y = max(0, center_y - search_range_y)
        # end_y = min(target_height, center_y + search_range_y)
        # start_x = max(0, center_x - search_range_x)
        # end_x = min(target_width, center_x + search_range_x)
        
        if scale == scale_range[0]:  # Print only for first scale
            print(f"Search area for scale {scale}:")
            print(f"X range: {start_x} to {end_x}")
            print(f"Y range: {start_y} to {end_y}")
            print(f"Scaled template size: {scaled_size}")
        
        # Slide the window around the center region
        for y in range(start_y, end_y + 1, step_size):
            for x in range(start_x, end_x + 1, step_size):
                similarity, scaled_template, region = calculate_similarity_at_position(
                    template, target, x, y, scale
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = region
                    best_template = scaled_template
                    best_position = (x, y)
                    best_scale = scale
                    
                    # Print when we find a better match
                    print(f"New best match: Similarity {similarity:.2f}%, "
                          f"Position: ({x}, {y}), Scale: {scale:.2f}")
                    
                    # Create and save the comparison image immediately
                    comparison = create_comparison_image(template, target, best_template, best_match, 
                                                      best_position, best_scale, best_similarity)
                    cv2.imwrite('edge_comparison.png', comparison)
    
    return best_similarity, best_template, best_match, best_position, best_scale

def resize_maintain_aspect(img, max_width, max_height):
    h, w = img.shape[:2]
    aspect = w / h
    if w > h:
        new_w = max_width
        new_h = int(new_w / aspect)
    else:
        new_h = max_height
        new_w = int(new_h * aspect)
    return cv2.resize(img, (new_w, new_h))

def calculate_overlap_similarity(img1, img2):
    _, img1_bin = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, img2_bin = cv2.threshold(img2, 127, cv2.THRESH_BINARY)
    overlap = cv2.bitwise_and(img1_bin, img2_bin)
    union = cv2.bitwise_or(img1_bin, img2_bin)
    if cv2.countNonZero(union) > 0:
        overlap_similarity = 100 * cv2.countNonZero(overlap) / cv2.countNonZero(union)
    else:
        overlap_similarity = 0
    return overlap_similarity

#比較画像読み込み
#実験
image1_path = '16.png'
#シミュレーション
image2_path = 'gravity.png'


# Load and process both images
contour1 = load_and_preprocess_image(image1_path, crop_ratio=(0.1, 0.9))#上下の切り取り範囲指定
contour2 = load_and_preprocess_image(image2_path)

template = cv2.imread(contour1)
target = cv2.imread(contour2)

# Print image sizes
print(f"Template size: {template.shape}")
print(f"Target size: {target.shape}")

# Find best match
print("\nFinding best match (this may take a while)...")
similarity, best_template, best_match, position, scale = find_best_match(
    template, target, 
    scale_range=(2.22, 2.30, 0.01),  #捜索するイメージサイズの範囲とステップ幅を設定
    step_size=1  # 移動ピクセル設定
)

print(f"Best match found:")
if similarity > 0:
    print(f"Similarity: {similarity:.2f}%")
    print(f"Position: {position}")
    print(f"Scale: {scale:.2f}")
else:
    print("No valid match found")
