restoredefaultpath;
clear all;close all;

% Replace this path with your cs4379c-fall2021 git repository path in your system.
repo_path = 'C:\School Txstate\Fall 2021\Computer Vision\cs4379c-fall2021\Adaboost';

s = filesep;

addpath([repo_path s 'training_test_data']);
addpath([repo_path s 'functions']);
addpath([repo_path s 'training_test_data' s 'training_nonfaces']);
addpath([repo_path s 'training_test_data' s 'training_faces']);
addpath([repo_path s 'training_test_data' s 'test_nonfaces']);
addpath([repo_path s 'training_test_data' s 'test_face_photos']);
addpath([repo_path s 'training_test_data' s 'test_cropped_faces']);

%%
% specify the extension of your image file
training_faces = dir('training_test_data/training_faces/*.bmp');  
size_training_faces = size(training_faces, 1);

% specify the extension of your image file
training_nonfaces = dir('training_test_data/training_nonfaces/*.jpg');  
size_training_nonfaces = size(training_nonfaces, 1);


%%%%Gather images from folder and store in array for both faces and
%%%%nonfaces, then compute integrals for each and store 
face_vertical = 50;
face_horizontal = 50;

total_faces_to_process = 500;

training_faces_cell = cell(size_training_faces, 1);
training_faces_arr = zeros(face_vertical, face_horizontal, total_faces_to_process);
training_faces_integrals = zeros(face_vertical, face_horizontal, total_faces_to_process);
for i = 1: total_faces_to_process
    crop = getfield(training_faces(i),'name');
    gray_Image = read_gray(crop);
    [x, y] = find(gray_Image);
    
    center_x = mean(x);
    center_y = mean(y);
    centroid = [center_x/2, center_y/2];
    
    training_sample = imcrop(gray_Image, [centroid 49 49]);
    training_faces_cell{i} = training_sample;
    
    training_faces_arr(:,:,i) = cell2mat(training_faces_cell(i));
    
    int_Image = imresize(gray_Image, [face_vertical face_horizontal]);
    B = integral_image(int_Image);
    training_faces_integrals(:,:,i) = B;
end

training_nonfaces_cell = cell(size_training_nonfaces, 1);
training_nonfaces_arr = zeros(face_vertical, face_horizontal, size_training_nonfaces);
training_nonfaces_integrals = zeros(face_vertical, face_horizontal, size_training_nonfaces);

for i = 1: size_training_nonfaces
    crop = getfield(training_nonfaces(i),'name');
    gray_Image = read_gray(crop);
    [x, y] = find(gray_Image);
    
    center_x = mean(x);
    center_y = mean(y);
    centroid = [center_x/2, center_y/2];
    
    training_sample = imcrop(gray_Image, [centroid 49 49]);
    training_nonfaces_cell{i} = training_sample;
    
    training_nonfaces_arr(:,:,i) = cell2mat(training_nonfaces_cell(i));
    
    int_Image = imresize(gray_Image, [face_vertical face_horizontal]);
    B = integral_image(int_Image);
    training_nonfaces_integrals(:,:,i) = B;
     
end


%%
% choosing a set of random weak classifiers

number = 1000;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end
save weak_classifiers_DB weak_classifiers
%%
%%%create labels and responces for adaboost 
example_number = size(training_faces_arr, 3) + size(training_nonfaces_arr, 3);
labels = zeros(example_number, 1);
labels (1:size(training_faces_arr, 3)) = 1;
labels((size(training_faces_arr, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(training_faces_arr, 3)) = training_faces_integrals;
examples(:, :, (size(training_faces_arr, 3)+1):example_number) = training_nonfaces_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end


%%
tic;
boosted_classifier = AdaBoost(responses, labels, 55)
toc;

save boosted boosted_classifier 
%%
%%%Bootstrapping 

samples(:, :, 1:total_faces_to_process) = training_faces_arr;
samples(:, :, total_faces_to_process+1:(total_faces_to_process + size_training_nonfaces)) ...
    = training_nonfaces_arr;



num_correct = []; num_incorrect = []; 


threshold = 5;
result_ARR = zeros(1,total_faces_to_process + size_training_nonfaces);
for i =1:(total_faces_to_process + size_training_nonfaces)
    
    photo = samples(:,:,i);
    imshow(samples(:,:,i), []);
    disp(i);
    
    
    result = boosted_predict(photo, boosted_classifier, weak_classifiers, 15);
    result_ARR(1, i) = result;

    label = labels(i,1);
    
    if (label == 1 && (result >= 0 || (result + threshold) >= 0))
        num_correct(end+1,1) = i;
  
    elseif (label == -1 && (result < 0 || (result + threshold) < 0))
        num_correct(end+1,1) = i;
   
    else
       num_incorrect(end+1,1) = i;
      
    end
                          
end

%%
%%%%Retrain mis-classifications 
retrain_correct_2 = []; retrain_incorrect_2 = [];


[r, c] = size(num_incorrect);

for i = 1: r
    index = num_incorrect(i, 1); 
    photo = samples(:,:,index);
    
    imshow(samples(:,:,index), []);
    disp(i);
    
    result = boosted_predict(photo, boosted_classifier, weak_classifiers, 15);
    
    result_ARR(1, i) = result;
    label = labels(i,1);
    
    if (label == 1 && (result >= 0 || (result + threshold) >= 0))
        retrain_correct_2(end+1,1) = i;
       
  
    elseif (label == -1 && (result < 0 || (result + threshold) < 0))
        retrain_correct_2(end+1,1) = i;
        
    else
       retrain_incorrect_2(end+1,1) = i;
       
    end
          
end




