function [average_face] = training_average_face()

test_face = dir('training_test_data/training_faces/*.bmp');

number1 = 870;
image_vertical = 2000;
image_horizontal = 1000;
total = zeros(image_vertical, image_horizontal);
for index = 1: number1
image = read_gray(test_face(index).name);
image = imresize(image, [2000, 1000]);
total = total + image;
end

totalFaces = total; 
test_face2 = dir('training_test_data/training_nonfaces/*.jpg');

number2 = 130;
image_vertical = 2000;
image_horizontal = 1000;
total = zeros(image_vertical, image_horizontal);
for index = 1: number2
image = read_gray(test_face2(index).name);
image = imresize(image, [2000, 1000]);
total = total + image;
end

average_face = (totalFaces + total) / (number1 + number2);

end

%xtrain_faces = faces(:, :, 1:700);
%xtrain_non = nonfaces(:, :, 1:700);

