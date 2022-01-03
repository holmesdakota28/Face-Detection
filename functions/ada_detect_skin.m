function skin_result = ada_detect_skin(pos_histogram, neg_histogram)

    load booted15
    load classifiers_weakDB1000
    
    num_results = 4;
    scales = 2;
    kernel_size = [75, 75];
    
    positive = pos_histogram;
    negative = neg_histogram;
    testing_faces = dir('training_test_data/test_face_photos/*.bmp'); 
    
    for index = 1:numel(testing_faces)
        rgb_img = getfield(testing_faces(index), 'name');
        rgb_double = read_gray(double(imread(resulting_img)));
        gray_img = read_gray(rgb_img);
     
      if(size(rgb_img)~= 3)
         disp('Malformed image, missing channel 3');        
      elseif(size(rgb_img, 3) == 3)
         skin_responce = detect_skin(rgb_double, positive, negative);
         
         boosted_responces = boosted_detector_demo(gray_img, gray_img, ...
            scales, boostedclassifer, classifiers_weak_list, kernel_size, num_results);
       end
    end
    
end