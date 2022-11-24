clear
clc

% load all the data
load("ZigZagVec.mat")   % 1 x 64 vector
load("cheetahMat.txt")  % our image
load("TrainingSamplesDCT_8_new.mat")    % the training data sets
load("zigzagImg.mat") % the preprocessed image
load("cheetah_mask.mat")    % the preprocessed mask

% dimensions for the training & the parameters for EM
C = 8; % number of components
dimensionList = [1,2,4,8,16,24,32,40,48,56,64]; % number of dimensions of the space

% # of samples
[n_cheetah, ~] = size(TrainsampleDCT_FG);
[n_grass, ~]   = size(TrainsampleDCT_BG);
totalSamples = (n_cheetah + n_grass);

% get the size of the image
[m, n] = size(cheetahMat);

% create a matrix
m = m - 7;
n = n - 7;
totalPixels = m * n;

% --- EM for cheetah ---

% initialize pi by creating a random normalized 1 x C matrix
pi_cheetah = randi(1, C);
pi_grass   = randi(1, C);
pi_cheetah = pi_cheetah / sum(pi_cheetah);
pi_grass   = pi_grass   / sum(pi_grass);

% initialize mu by choosing C samples randomly from the training data set
p = randperm(n_cheetah, C);    % pick C random numbers from 0 to the size of the sample of cheetah
mu_cheetah = zeros(C, 64);  % create an array for mu
mu_grass   = zeros(C, 64);  % create an array for mu
for i = 1 : C
    mu_cheetah(i, :) = TrainsampleDCT_FG(p(i), 64);
    mu_grass(i, :)   = TrainsampleDCT_BG(p(i), 64);
end

% initialize sigma by creating a diagonal matrix with random variable
sigma_cheetah = zeros(64, 64, C);
sigma_grass   = zeros(64, 64, C);
for i = 1 : C
    randomVec = rand(1, 64);    % generate a random vector (1 x 64)
    sigma_cheetah(:, :, i) = diag(randomVec);   % diagonlize the random vector
    randomVec = rand(1, 64);    % generate a random vector (1 x 64)
    sigma_grass(:, :, i) = diag(randomVec);   % diagonlize the random vector
end

EMLimit = 1000; % maximum iterations

% --- Start of EM for cheetah ---
jointPro_cheetah = zeros(n_cheetah, C);
for i = 1 : EMLimit
    % E step
    for j = 1 : C
        % P_{X, Z}(x, z; psi) = P_{X|Z}(x|z; psi) * P_Z(z; psi)
        jointPro_cheetah(:, j) = mvnpdf(TrainsampleDCT_FG, mu_cheetah(j, :), sigma_cheetah(:, :, j)) * pi_cheetah(j); % mvnpdf returns the pdf of given X, mu, sigma 
    end

    % calculate the log likelihood of the
    sumRow = sum(jointPro_cheetah, 2);    % returns the sum of the elements in each row 
    likelihood_cheetah = sum(log(sumRow));

    % h_ij = (G(x_i, u_j, sigma_j) * pi_j) / sum(G(x_i, u_j, sigma_j) * pi_j)
    hij_cheetah = jointPro_cheetah ./ sumRow;

    % M step

    % update pi for (n+1)
    % pi = 1 / n * (sum(hij))
    pi_cheetah = sum(hij_cheetah) / n_cheetah;

    % update mu for (n+1)
    % mu = sum(hij * x)/sum(hij)
    % TODO: mu_cheetah changes dimension
    mu_cheetah = (hij_cheetah' * TrainsampleDCT_FG) ./ sum(hij_cheetah)';

    % update sigma for (n+1)
    % sigma^2 = sum(hij * (x - mu)^2) / sum(hij)
    for j = 1:C
        sigma_cheetah(:,:,j) = diag(diag(((TrainsampleDCT_FG - mu_cheetah(j,:))'.*hij_cheetah(:,j)'* ... 
            (TrainsampleDCT_FG - mu_cheetah(j,:)) ./ sum(hij_cheetah(:,j),1)) + 0.0000001));
    end

    % break condition
    % breaks the loop if the likelihood does not change more than 0.001
    if i > 1
        if abs(likelihood_cheetah - likelihood_cheetah_previous) < 0.001
            break; 
        end
    end
    
    % store the current result as the previous
    likelihood_cheetah_previous = likelihood_cheetah;

end
% --- End of EM for cheetah ---

% --- Start of EM for grass ---
jointPro = zeros(n_grass, C);
for i = 1 : EMLimit
    % E step
    for j = 1 : C
        % P_{X, Z}(x, z; psi) = P_{X|Z}(x|z; psi) * P_Z(z; psi)
        jointPro(:, j) = mvnpdf(TrainsampleDCT_BG, mu_grass(j, :), sigma_grass(:, :, j)) * pi_grass(j); % mvnpdf returns the pdf of given X, mu, sigma 
    end

    % calculate the log likelihood of the
    sumRow = sum(jointPro, 2);    % returns the sum of the elements in each row 
    likelihood_grass = sum(log(sumRow));

    % h_ij = (G(x_i, u_j, sigma_j) * pi_j) / sum(G(x_i, u_j, sigma_j) * pi_j)
    hij_grass = jointPro ./ sumRow;

    % M step

    % update pi for (n+1)
    % pi = 1 / n * (sum(hij))
    pi_grass = sum(hij_grass) / n_grass;

    % update mu for (n+1)
    % mu = sum(hij * x)/sum(hij)
    mu_grass = hij_grass' * TrainsampleDCT_BG ./ sum(hij_grass)';

    % update sigma for (n+1)
    % sigma^2 = sum(hij * (x - mu)^2) / sum(hij)
    for j = 1:C
        sigma_grass(:, :, j) = diag(diag(((TrainsampleDCT_BG - mu_grass(j,:))'.*hij_grass(:,j)'* ...
            (TrainsampleDCT_BG - mu_grass(j,:)) ./ sum(hij_grass(:,j),1)) + 0.0000001));
    end

    % break condition
    % breaks the loop if the likelihood does not change more than 0.001
    if i > 1
        if abs(likelihood_grass - likelihood_grass_previous) < 0.001
            break; 
        end
    end

    % store the current result as the previous
    likelihood_grass_previous = likelihood_grass;

end
% --- End of EM for grass ---

% --- BDR ---
lenList = length(dimensionList);
errorMat = zeros(1, lenList);
for curDim = 1 : lenList

    Kth = dimensionList(curDim);

    % compare BDR for EM
    maskVec = zeros(m * n, 1);   % a vector of our mask, will resize it later

    for x = 1 : length(zigzagImg)
        
        % set the probability of each class as zero
        pro_cheetah = 0;
        pro_grass   = 0;

        % compute total BDR for cheetah
        for y = 1:size(mu_cheetah,1)
            pro_cheetah = pro_cheetah + mvnpdf(zigzagImg(x, 1 : Kth), mu_cheetah(y, 1 : Kth), sigma_cheetah(1 : Kth, 1 : Kth,y)) * pi_cheetah(y);
        end

        % compute total BDR for grass
        for y = 1:size(mu_grass,1)
            pro_grass = pro_grass + mvnpdf(zigzagImg(x, 1 : Kth), mu_grass(y, 1 : Kth),sigma_grass(1 : Kth, 1 : Kth,y))*pi_grass(y);
        end
        
        % decide whether the pixel is cheetah or grass
        if pro_cheetah > pro_grass
            maskVec(x) = 1;
        end
    end

    % resize the vector to matrix
    maskMat = Vec2Mat(maskVec);

    % compute the error rate
    errorMat(curDim) = Err(cheetah_mask, maskMat);
    errorRate = errorMat ./ (m * n);
    imshow(maskMat)
    figure
end