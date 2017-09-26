function densecrf_preprocess(model_type)
  if strcmp(model_type, 'appearance')
    data_dir = './images/';
    input_file = './appearance_image_list.txt';
    output_file = './appearance_output_list.txt';
  elseif strcmp(model_type, 'motion')
    data_dir = './motion_images/';
    input_file = './motion_image_list.txt';
    output_file = './motion_output_list.txt';
  end

  image_files    = textread(input_file,'%s');
  image_prefixes = textread(output_file,'%s');
  num_images = length(image_files);

  for i = 1:num_images
    feature_name = [image_prefixes{i} '_blob_0.mat'];
    data = load(fullfile(data_dir, feature_name));
    raw_result = permute(data.data, [2 1 3]);

    img = imread(fullfile(data_dir,image_files{i}));
    img_row = min(size(img, 1),size(raw_result,1));
    img_col = min(size(img, 2),size(raw_result,2));

    probs = raw_result(1:img_row, 1:img_col, :);
    file_name = fullfile(data_dir, [image_prefixes{i} '_anno.ppm']);
    imwrite(probs(:,:,2), file_name);

    imwrite(img, fullfile(data_dir, [image_prefixes{i} '.ppm']));
  end
