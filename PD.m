clear
clc

% --- PE METHOD ---

% load the data
load("Alpha.mat");  % 1 x 9 vector
load("Prior_2.mat");    % Strategy 1 (W0 mu0_FG mu0_BG) %TODO: change
load("TrainingSamplesDCT_subsets_8.mat", "D4_BG", "D4_FG");   % D1: cheetah & grass
load("ZigZagVec.mat")   % 1 x 64 vector
load("cheetahMat.txt")  % our image
load("cheetah_mask.mat")    % our ideal mask

% get the size of the image
[m, n] = size(cheetahMat);

% create a matrix
m = m - 7;
n = n - 7;
totalPixels = m * n;

% # of samples
[n_cheetah, ~] = size(D4_FG);
[n_grass, ~]   = size(D4_BG);
totalSamples = (n_cheetah + n_grass);

% calculate the priors
Prior_cheetah = n_cheetah / totalSamples;
Prior_grass   = n_grass   / totalSamples;

% calculate the mu_MLs
mu_ML_cheetah = mean(D4_FG);
mu_ML_grass   = mean(D4_BG);

% calculate the covs
cov_cheetah = cov(D4_FG);
cov_grass   = cov(D4_BG);

% results
errorPE_D4P2 = zeros(1, 9);  %TODO: change

% Prior 1
for d = 1 : 9
    % calculate the sigma_0
    sigma_0 = diag(alpha(d) .* W0);
        
    % calculate the mu_ns
    mu_n_cheetah = ((n_cheetah .* sigma_0) ./ (cov_cheetah + n_cheetah .* sigma_0)) .* mu_ML_cheetah + ((cov_cheetah) ./ (cov_cheetah + n_cheetah .* sigma_0)) .* mu0_FG;
    mu_n_grass   = ((n_grass   .* sigma_0) ./ (cov_grass   + n_grass   .* sigma_0)) .* mu_ML_grass   + ((cov_grass)   ./ (cov_grass   + n_grass   .* sigma_0)) .* mu0_BG;
    
    % calculate the sigma_ns
    sigma_n_cheetah = 1 ./ ((1 ./ cov_cheetah) + (n_cheetah ./ cov_cheetah));
    sigma_n_grass   = 1 ./ ((1 ./ cov_grass)   + (n_grass   ./ cov_grass));
    
    % calculate SIGMAs (sigma + sigma_n) and their inverses
    SIGMA_cheetah = sigma_n_cheetah + cov_cheetah;
    SIGMA_grass   = sigma_n_grass   + cov_grass;
    SIGMAInv_cheetah = inv(SIGMA_cheetah);
    SIGMAInv_grass   = inv(SIGMA_grass);
    
    % calculate the coefficients
    deno_cheetah = (sqrt(((2 * pi) ^ 64) * det(SIGMA_cheetah)));
    deno_grass   = (sqrt(((2 * pi) ^ 64) * det(SIGMA_grass)));
    
    % calculate DCT2 (64D)
    error = 0;  % reset the error
    for i = 1 : m
        parfor j = 1 : n
            Block = cheetahMat(i : i + 7, j : j + 7);
            Block_DCT = dct2(Block, 8, 8);
            V = Block_DCT(:).';
            X = zeros(1, 64);
            % mapping
            for k = 1 : 64
                X(ZigZagVec(k)) = V(k);
            end
    
            % calculate the i*(x)
            P_X_likelihood_cheetah = exp(-0.5 .* (X - mu_n_cheetah) * (SIGMAInv_cheetah) * (X - mu_n_cheetah).') ./ deno_cheetah;
            P_X_likelihood_grass   = exp(-0.5 .* (X - mu_n_grass) * (SIGMAInv_grass)     * (X - mu_n_grass).')   ./ deno_grass;
            P_X_cheetah = P_X_likelihood_cheetah * Prior_cheetah;
            P_X_grass   = P_X_likelihood_grass   * Prior_grass;
    
            % based on the results, decide whether the pixel is grass or cheetah
            if (P_X_grass > P_X_cheetah)
                result = 0;
            else
                result = 1;
            end

            % compare my result with the ideal mask
            if (result ~= cheetah_mask(i, j))
                error = error + 1;
            end
        end
    end
    errorPE_D4P2(1, d) = error / totalPixels; 
end
            