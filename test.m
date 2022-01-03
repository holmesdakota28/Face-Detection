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
%%Load in data from folders
load boosted;
load weak_classifiers_DB;

test_faces_images = dir('training_test_data/test_face_photos/*.jpg');  
size_test_faces = size(test_faces_images, 1);

test_nonfaces_images = dir('training_test_data/test_nonfaces/*.jpg');  
size_test_nonfaces = size(test_nonfaces_images, 1);

test_croppedfaces_images = dir('training_test_data/test_croppedfaces/*.bmp');  
size_test_croppedfaces = size(test_croppedfaces_images, 1);
%%
face_vertical = 100;
face_horizontal = 100;


test_faces_arr = zeros(face_vertical, face_horizontal, 3, size_test_faces);
for i = 1: size_test_faces
    crop = getfield(test_faces_images(i),'name');
    gray_Image = imread(crop);
    temp_Image = imresize(gray_Image, [face_vertical face_horizontal]);
    test_faces_arr(:,:,:,i) = temp_Image;
end

test_nonfaces_arr = zeros(face_vertical, face_horizontal, 3, size_test_nonfaces);
for i = 1: size_test_nonfaces
    crop = getfield(test_nonfaces_images(i),'name');
    gray_Image = imread(crop);
    temp_Image = imresize(gray_Image, [face_vertical face_horizontal]);
    test_nonfaces_arr(:,:,:,i) = temp_Image;
end

test_croppedfaces_arr = zeros(face_vertical, face_horizontal, 3, size_test_croppedfaces);
for i = 1: size_test_croppedfaces
    crop = getfield(test_croppedfaces_images(i),'name');
    gray_Image = imread(crop);
    temp_Image = imresize(gray_Image, [face_vertical face_horizontal]);
    test_croppedfaces_arr(:,:,:,i) = temp_Image;
end
%%
%%%%Skin detection on test_faces_images
%%%%Just like in train.m with face integrals, compute skin integrals, then
%%%%combine with classifiers to create responces for adaboost function.
%%%%then re-run adaboost, obtain result and pass into boosted detector demo

negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

skin_integrals = zeros(face_vertical, face_horizontal, size_test_faces);
for i = 1; size_test_faces
    crop = getfield(test_faces_images(i),'name');
    gray_Image = read_gray(crop);
    int_Image = imresize(gray_Image, [face_vertical face_horizontal]);
    B = integral_image(int_Image);
    skin_integrals(:,:,i) = B;
end

examples = zeros(face_vertical, face_horizontal, size_test_faces, size_test_faces);
examples (:, :, :, 1:size(test_faces_arr, size_test_faces)) = skin_integrals;

classifier_number = numel(weak_classifiers);
responses =  zeros(classifier_number, 20);

for example = 1:size_test_faces
    integral = examples(:, :, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end
labels = zeros(20, 1);
tic;
skin_boosted_classifier = AdaBoost(responses, labels, 55)
toc;

for i = 1: size_test_faces
    boosted_Image = read_gray(getfield(test_faces_images(i),'name'));
    
    boosted_result = boosted_detector_demo(boosted_Image, 2,...
        skin_boosted_classifier, weak_classifiers, [50,50], 2);
    figure; 
    imshow(boosted_result, []);
end

%%



