import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def load_and_preprocess_image(image_path, crop_ratio=None):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Split the channels and enhance green channel detection
    b, g, r = cv2.split(img)
    
    # Create a mask for green pixels
    green_mask = cv2.inRange(img, (0, 50, 0), (50, 255, 50))
    
    # Combine green mask with grayscale for better edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    combined = cv2.addWeighted(gray, 0.5, green_mask, 0.5, 0)
    
    # Apply Gaussian blur with smaller kernel
    blurred = cv2.GaussianBlur(combined, (3, 3), 0)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Use Canny edge detection with adjusted thresholds
    edges = cv2.Canny(enhanced, 0, 80)  # Lower thresholds for better edge detection
    
    # Dilate edges to connect gaps
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours with different method
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

def calculate_similarity(template_gray, target_gray):
    # Ensure images are grayscale
    if len(template_gray.shape) > 2:
        template_gray = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)
    if len(target_gray.shape) > 2:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
    
    # Convert to binary
    _, template_bin = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)
    _, target_bin = cv2.threshold(target_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate SSIM
    ssim_score, _ = ssim(template_gray, target_gray, full=True)
    ssim_similarity = max(0, min(100, (ssim_score + 1) * 50))  # Convert from [-1,1] to [0,100]
    
    # Calculate area similarity
    template_area = cv2.countNonZero(template_bin)
    target_area = cv2.countNonZero(target_bin)
    area_ratio = min(template_area, target_area) / max(template_area, target_area) if max(template_area, target_area) > 0 else 0
    area_similarity = min(100, max(0, area_ratio * 100))
    
    # Calculate perimeter similarity
    template_contours, _ = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    template_perimeter = sum(cv2.arcLength(cnt, True) for cnt in template_contours) if template_contours else 0
    target_perimeter = sum(cv2.arcLength(cnt, True) for cnt in target_contours) if target_contours else 0
    
    perimeter_ratio = min(template_perimeter, target_perimeter) / max(template_perimeter, target_perimeter) if max(template_perimeter, target_perimeter) > 0 else 0
    perimeter_similarity = min(100, max(0, perimeter_ratio * 100))
    
    # Calculate overlap similarity
    kernel = np.ones((3,3), np.uint8)
    template_dilated = cv2.dilate(template_bin, kernel, iterations=1)
    target_dilated = cv2.dilate(target_bin, kernel, iterations=1)
    
    # Debug: Save the dilated images to see what they look like
    cv2.imwrite("debug_template_dilated.png", template_dilated)
    cv2.imwrite("debug_target_dilated.png", target_dilated)
    
    # Use dilated images for both overlap and union calculations
    overlap = cv2.bitwise_and(template_dilated, target_dilated)
    union = cv2.bitwise_or(template_dilated, target_dilated)
    
    # Debug: Save overlap and union images
    cv2.imwrite("debug_overlap.png", overlap)
    cv2.imwrite("debug_union.png", union)
    
    overlap_count = cv2.countNonZero(overlap)
    union_count = cv2.countNonZero(template_dilated)
    
    # print(f"Debug - Overlap pixels: {overlap_count}")
    # print(f"Debug - Union pixels: {union_count}")
    
    # Ensure overlap is truly zero if there's no intersection
    if overlap_count < 10:  # Allow for some noise
        overlap_similarity = 0
    else:
        overlap_similarity = min(100, max(0, 100 * overlap_count / union_count))

    # Combine all similarity metrics with adjusted weights
    #類似度の測定値の比重設定
    similarity = (
        area_similarity * 0.1 +      # Increased weight for area
        perimeter_similarity * 0.1 +  # Moderate weight for perimeter
        ssim_similarity * 0.5 +       # Moderate weight for SSIM
        overlap_similarity * 0.3      # Increased weight for overlap
    )

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

def find_best_match(template, target, scale_range, step_size=1, search_range=(0.4,0.6)):
    """Find the best matching position and scale of template within target"""
    best_similarity = 0
    best_match = None
    best_template = None
    best_position = None
    best_scale = None
    
    target_height, target_width = target.shape[:2]
    template_height, template_width = template.shape[:2]
    
    print(f"\nTemplate original size: {template_width}x{template_height}")
    print(f"Target size: {target_width}x{target_height}")
    
    # Calculate center position
    center_y = target_height // 2
    center_x = target_width // 2
    
    # Calculate search range as percentage of target dimensions
    search_range_y = int(target_height * search_range[1])  # 40% of target height
    search_range_x = int(target_width * search_range[0])   # 40% of target width
    
    for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]):
        print(f"\nChecking scale: {scale}")
        # Scale template
        scaled_size = (int(template_width * scale), int(template_height * scale))
        print(f"Scaled template size: {scaled_size[0]}x{scaled_size[1]}")
        
        # Skip if scaled template is larger than target
        if scaled_size[0] > target_width or scaled_size[1] > target_height:
            print(f"Skipping scale {scale} - template too large for target")
            print(f"Scaled size: {scaled_size[0]}x{scaled_size[1]} vs Target: {target_width}x{target_height}")
            continue
        
        # Calculate desired search area around center
        desired_start_y = max(0, center_y - search_range_y)
        desired_end_y = min(target_height, center_y + search_range_y)
        desired_start_x = max(0, center_x - search_range_x)
        desired_end_x = min(target_width, center_x + search_range_x)
        
        # Adjust search boundaries to fit within target image and account for template size
        start_y = max(0, min(desired_start_y, target_height - scaled_size[1]))
        end_y = min(target_height - scaled_size[1], desired_end_y)
        start_x = max(0, min(desired_start_x, target_width - scaled_size[0]))
        end_x = min(target_width - scaled_size[0], desired_end_x)
        
        print(f"Search area for scale {scale}:")
        print(f"X range: {start_x} to {end_x}")
        print(f"Y range: {start_y} to {end_y}")
        
        # Check if we have a valid search area
        if start_y > end_y or start_x > end_x:
            print(f"No valid search area available at this scale")
            continue
        
        # Slide the window around the center region
        for y in range(int(start_y), int(end_y) + 1, step_size):
            for x in range(int(start_x), int(end_x) + 1, step_size):
                similarity, scaled_template, region = calculate_similarity_at_position(
                    template, target, x, y, scale
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = region
                    best_template = scaled_template
                    best_position = (x, y)
                    best_scale = scale
                    
                    # Get individual similarity scores
                    # Convert to grayscale if needed and calculate appropriate window size
                    template_gray = cv2.cvtColor(best_template, cv2.COLOR_BGR2GRAY) if len(best_template.shape) == 3 else best_template
                    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                    
                    min_dim = min(template_gray.shape[0], template_gray.shape[1])
                    win_size = min(7, min_dim - (min_dim % 2 == 0))  # Ensure it's odd and not larger than image
                    if win_size < 3:
                        win_size = 3  # Minimum window size of 3
                    
                    try:
                        ssim_score = ssim(template_gray, region_gray, win_size=win_size, data_range=255)
                    except ValueError:
                        # If SSIM fails, use a simpler metric
                        ssim_score = 0
                        print("Warning: SSIM calculation failed, using 0 as score")
                    
                    overlap_score = calculate_overlap_similarity()
                    
                    print(f"\nNew best match found!")
                    print(f"Overall Similarity: {similarity:.2f}%")
                    print(f"SSIM Score: {ssim_score:.4f}")
                    print(f"Overlap Score: {overlap_score:.2f}%")
                    print(f"Position: ({x}, {y})")
                    print(f"Scale: {scale:.2f}")
                    
                    comparison = create_comparison_image(template, target, best_template, best_match, 
                                                      best_position, best_scale, best_similarity)
                    cv2.imwrite('edge_comparison.png', comparison)
                    if similarity == 100:
                        return 100, best_template, best_match, best_position, best_scale
        center_x = best_position[0]
        center_y = best_position[1]
    
    if best_position is None:
        print("\nNo valid match found - all scales resulted in template too large or invalid search area")
    else:
        print(f"\nBest match found at scale {best_scale:.2f}")
        print(f"Position: ({best_position[0]}, {best_position[1]})")
        print(f"Similarity: {best_similarity:.2f}%")
    
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

def calculate_overlap_similarity():
    overlap = cv2.imread("debug_overlap.png", cv2.IMREAD_GRAYSCALE)
    union = cv2.imread('debug_template_dilated.png', cv2.IMREAD_GRAYSCALE)
    if cv2.countNonZero(union) > 0:
        overlap_similarity = 100 * cv2.countNonZero(overlap) / cv2.countNonZero(union)
    else:
        overlap_similarity = 0
    return overlap_similarity

#比較画像読み込み
#実験
image1_path = '17.png'
#シミュレーション
image2_path = '52.png'

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

# Calculate base scale from image sizes
template_size = max(template.shape[0], template.shape[1])
target_size = max(target.shape[0], target.shape[1])
base_scale = target_size / template_size

# Set scale range to ±10% of base scale
# 自動計算された画像のスケールレンジ
min_scale = base_scale * 0.8  # -20%
max_scale = base_scale * 1.2  # +20%
step = (max_scale - min_scale) / 50  # 10 steps across the range

similarity, best_template, best_match, position, scale = find_best_match(
    template, target, 
    scale_range=(min_scale, max_scale, step),  # Automatically calculated scale range
    step_size=5 , # 移動ピクセル設定
    search_range=(0.4,0.6) # 検索範囲設定
)

print(f"Best match found:")
if similarity > 0:
    print(f"Similarity: {similarity:.2f}%")
    print(f"Position: {position}")
    print(f"Scale: {scale:.2f}")
else:
    print("No valid match found")
