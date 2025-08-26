% This script retrieves scattered fields (E_ret) using image_data.mat
% image_date.mat includes the camera image (image_data) and corresponding SLM patterns (slmPattern)
% image_data vairable contains background subtracted camera images converted into single precision.
% slmPattern variable contains binary phase pattern displayed on the SLM.
%
% 'field_data_spiral.mat' provides one example of retrieved fields E_ret.
% 
% Requires MATLAB 2023a or later, due to the MATLAB function "random". Otherwise, Statistics and Machine Learning Toolbox should be installed. 
% Written by YoonSeok Baek

%% Initialization
% load image data for the spiral shaped object

clear
dataName = 'image_data.mat'; 

script_dir = fileparts(matlab.desktop.editor.getActiveFilename);
cd(script_dir);

if ~isfile(dataName)
    disp('Cannot find the data file. Open it manually')
    [~,dataPath] = uigetfile('*.mat','open image_data.mat');
    cd(dataPath)
end

load(dataName)

image_count = size(image_data,3);

% Define setup paramters
d_x = 20; % pixel size at the SLM plane.
lambda = 0.532; % wavelength
d_k = 6.5*2; % camera pixel after binning
mag = 2; % magnification of the SLM by 4f
f = 250*10^3; % focal length of a Fourier transform lens
Nk = round(f*lambda/(d_x*mag)/d_k);

[dcy, dcx] = find(mean(image_data,3) > 10);
dcy = round(mean(dcy)); dcx = round(mean(dcx));

intensity = crop_image(single(image_data), Nk, [dcy, dcx]);

slm_size = size(slmPattern,[1,2]);
slm = single(exp(1i*double(logical(slmPattern))*pi));
slm = crop_image(slm,Nk,[-34,20]+floor(slm_size/2+1)); % adjust SLM pattern to the beam center

%% Iteration
% Apply the mixed state phase retrieval algorithm according to the Method Section (Field retrieval)

initialization_num = 20; 
iteration_max = 100; 
mode_num = 13; % number of modes to retrieve. should be large enough to give stable solutions. 

% Initialization; see also [Baek, Y., de Aguiar, H. B., & Gigan, S. (2023). Phase conjugation with spatially incoherent light in complex media. Nature Photonics, 17(12), 1114-1119.]
disp('Initalization starts')
E_ret = single(ones(Nk,Nk,mode_num).*exp(1i*random('uniform',-pi,pi,Nk,Nk,mode_num))); % random guess of fields at the SLM plane.
for ii = 1:initialization_num 
    for mm = 1:image_count
        psi = FFT(E_ret.*slm(:,:,mm));
        psi = intensity(:,:,mm)./(sqrt(sum(abs(psi).^2,3))+eps).*psi;             
        E_ret = conj(slm(:,:,mm)).*IFFT(psi);
    end
end

% Iterative phase retrieval 
error_list = zeros(iteration_max,1);
error_value = zeros(image_count,1);

disp('Phase retrieval starts')
for ii = 1:iteration_max 
    if round(ii/10) == ii/10
        disp(['iteration: ', num2str(ii),' / ',num2str(iteration_max)])
    end
    for mm = 1:image_count 
        psi = FFT(E_ret.*slm(:,:,mm));

        error_value(mm) = mean((intensity(:,:,mm)-sum(abs(psi).^2,3)).^2,'all'); 

        psi = sqrt(intensity(:,:,mm))./(sqrt(sum(abs(psi).^2,3))+eps).*psi;
        E_ret = conj(slm(:,:,mm)).*IFFT(psi);
    end

    error_list(ii) = mean(error_value); 
    if ii>10 && abs(error_list(ii)-error_list(ii-1))/error_list(ii) < 10^-4 % threshold to stop the iteration
        error_list = error_list(1:ii);
        break
    end
end
disp('Finished')
figure(13), plot(log10(error_list)), xlabel('Iteration'), ylabel('Error'), axis square

E_ret = FFT(E_ret); % field at the camera plane

%% Plot results
figure(14), clf, axis off
sgtitle('retrieved fields (only the first 6 are shown)')

for ii = 1:min(mode_num,6)
    test = E_ret(:,:,ii);
    subplot(2,min(mode_num,6),ii), imagesc(abs(E_ret(:,:,ii))), axis image off
    subplot(2,min(mode_num,6),min(mode_num,6)+ii), imagesc(angle(E_ret(:,:,ii))), axis image off
end

%% Custum functions
function out = FFT(in)
    out = fftshift(fft2(ifftshift(in)));
end

function out = IFFT(in)
    out = fftshift(ifft2(ifftshift(in)));
end

function data = crop_image(data,crop_size,centerArray)

    sizeArray = [crop_size, crop_size];

    centerArray = centerArray(:);
    data = data(centerArray(1)-floor(sizeArray(1)/2):centerArray(1)+ceil(sizeArray(1)/2)-1, ...
        centerArray(2)-floor(sizeArray(2)/2):centerArray(2)+ceil(sizeArray(2)/2)-1,:);
end



