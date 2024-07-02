def sort_boxes(boxes, image, percentage_threshold=0.05):
    # Calculate center coordinates for each box
    def center(box):
        x_min, y_min, x_max, y_max = box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return center_x, center_y

    # First, sort by center_y (vertical center)
    boxes = sorted(boxes, key=lambda box: center(box)[1])

    image_height, image_width = image.shape[:2]
    threshold = percentage_threshold * image_height

    # Sort each group of boxes that are close vertically by center_x (horizontal center)
    def vertical_group_sort(boxes, threshold):
        sorted_boxes = []
        line_indices = []
        current_group = [boxes[0]]
        current_line_index = 0

        for box in boxes[1:]:
            if abs(center(box)[1] - center(current_group[-1])[1]) <= threshold:
                current_group.append(box)
            else:
                # Sort the current group by center_x and add to sorted_boxes
                sorted_boxes.extend(sorted(current_group, key=lambda b: center(b)[0]))
                line_indices.extend([current_line_index] * len(current_group))
                current_line_index += 1
                current_group = [box]

        # Sort the last group by center_x and add to sorted_boxes
        sorted_boxes.extend(sorted(current_group, key=lambda b: center(b)[0]))
        line_indices.extend([current_line_index] * len(current_group))

        return sorted_boxes, line_indices

    return vertical_group_sort(boxes, threshold)

# Compute the area of a box
def compute_area(box):
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)

# Check if one box is overlapping another
def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max

# Filter out overlapping boxes
def filter_overlapping_boxes(boxes):
    filtered_boxes = []
    areas = [compute_area(box) for box in boxes]

    for i, box in enumerate(boxes):
        is_largest = True
        for j, other_box in enumerate(boxes):
            if i != j and is_overlapping(box, other_box):
                if areas[i] <= areas[j]:
                    is_largest = False
                    break
        if is_largest:
            filtered_boxes.append(box)

    return filtered_boxes

# Filter out small boxes based on a threshold percentage of the image area
def filter_small_boxes(boxes, image, area_percentage_threshold=0.001):
    image_height, image_width = image.shape[:2]
    area_threshold = area_percentage_threshold * (image_height * image_width)
    return [box for box in boxes if compute_area(box) >= area_threshold]

# Add buffer to each box to include some surrounding area
def add_box_buffer(boxes, image, area_percentage_threshold=0.0002):
    buffered_boxes = []
    image_height, image_width = image.shape[:2]
    BUFFERX = area_percentage_threshold * (image_height * image_width) / 100
    BUFFERY = area_percentage_threshold * (image_height * image_width) / 50
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        # Add buffer to each side of the box
        x_min -= BUFFERX
        y_min -= BUFFERY
        x_max += BUFFERX
        y_max += BUFFERY
        # Ensure the coordinates are within the image bounds
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, image_width - 1)
        y_max = min(y_max, image_height - 1)
        buffered_boxes.append([x_min, y_min, x_max, y_max])
    return buffered_boxes