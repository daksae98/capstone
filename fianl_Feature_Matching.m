%%%%%%%%%%%%%%%%%%%%%% Feature Extraction & Matching %%%%%%%%%%%%%%%%%%%%%%
%%% - Data loading                                                      %%%
%%% - Parameters set/load                                               %%%
%%% - Image Set recognition                                             %%%
%%% - Feature extraction                                                %%%
%%% - Feature matching                                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Created by...                                                       %%%
%%% - KangHyeok Choi.(cwsurgy@inha.ac.kr)                               %%%
%%% - 이름(이메일)                                                       %%%
%%% - 이름(이메일)                                                       %%%
%%% - 이름(이메일)                                                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Updated...                                                          %%%
%%% - 03.16.2023 (by K.H. Choi)                                         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Opening... 워크 스페이스의 데이터 삭제 및 명령 창 내용 삭제
clear
clc

%% Data Load... 이미지 데이터 경로 지정

% 이미지들의 경로와 이름을 path_img 및 list_img 변수에 저장
[list_img, path_img] = uigetfile({'*.*','image'},...
    'MultiSelect', 'on', sprintf('IMAGE DATA LOAD'));
list_img = list_img';

% 이미지들의 경로와 이름을 합쳐서 list_img에 저장 (이미지 불러오기의 편의성을 위해서...)
for numb_img = 1:size(list_img,1)
    if ischar(list_img) == 1
        file1 = cell2mat(list_img(numb_img,:));
    elseif iscell(list_img) == 1
        file1 = cell2mat(list_img(numb_img,1));
    end
    list_img{numb_img,1} = sprintf('%s%s', path_img, file1);
end

%% Parameter Set
scalef = 0.1;


%% Image Pair Set...
% 일단... 직접 입력(수동) 또는 이미지 순서대로 2장씩 자동으로 매칭쌍을 만드는 방식으로 작성
% 추후 자동화 + App형식으로 변경

% 수동1: image_pair = [1 2; 2 3; 3 4];
% 수동2: image_pair = [1 2; 1 3; 1 4];
% 자동1: image_pair = nchoosek(1:size(list_img,1),2); %>>> 모든 이미지 조합에 대하여 매칭 수행
image_pair = [(1:size(list_img,1)-1)' (2:size(list_img,1))']; % 자동2: 연속된 이미지에서만 매칭 수행

%% Feature Extraction
clc
close all

name_alg={'BRISK' 'FAST' 'KAZE' 'ORB' 'SIFT' 'SURF'};
ext_F = cell([6 size(list_img,1)]);    % 특징정보 저장을 위한 빈 행렬
ext_FPts = cell([6 size(list_img,1)]); % 특징점 좌표 저장을 위한 빈 행렬
ext_FT = NaN(6, size(list_img,1));   % 특징점 추출에 걸리는 시간 저장을 위한 빈 행렬

for numb_img = 1:size(list_img,1)

    % 이미지 불러오기
    filename1 = list_img{numb_img,1}; % 이미지 경로/이름
    fr1 = imread(filename1); % 이미지 불러 오기
    fr1 = imresize(fr1, scalef); % 이미지 크기 조정
    img1 = rgb2gray(fr1); % 특징점 추출을 위하여 이미지를 전정색(회색조)으로 변환


    % 알고리즘은 알파벳 순서...
    % BRISK 특징점 추출
    tic
    points1 = detectBRISKFeatures(img1);
    [features1, valid_points1] = extractFeatures(img1, points1, 'Method', 'BRISK');
    ext_FT(1,numb_img) = toc;
    ext_F{1,numb_img} = features1;
    ext_FPts{1,numb_img} = valid_points1;
    % Feature_ext{3,numb_img} = features;

    % FAST 특징점 추출
    tic
    points2 = detectFASTFeatures(img1);
    [features2, valid_points2] = extractFeatures(img1, points2);
    ext_FT(2,numb_img) = toc;
    ext_F{2,numb_img} = features2;
    ext_FPts{2,numb_img} = valid_points2;

    % KAZE 특징점 추출
    tic
    points3 = detectKAZEFeatures(img1);
    [features3, valid_points3] = extractFeatures(img1, points3, 'Method', 'KAZE');
    ext_FT(3,numb_img) = toc;
    ext_F{3,numb_img} = features3;
    ext_FPts{3,numb_img} = valid_points3;

    % ORB 특징점 추출
    tic
    points4 = detectORBFeatures(img1);
    [features4, valid_points4] = extractFeatures(img1, points4, 'Method', 'ORB');
    ext_FT(4,numb_img) = toc;
    ext_F{4,numb_img} = features4;
    ext_FPts{4,numb_img} = valid_points4;

    % SIFT 특징점 추출
    points5 = detectSIFTFeatures(img1);
    [features5, valid_points5] = extractFeatures(img1, points5, 'Method', 'SIFT');
    ext_FT(5,numb_img) = toc;
    ext_F{5,numb_img} = features5;
    ext_FPts{5,numb_img} = valid_points5;

    % SURF 특징점 추출
    tic
    points6 = detectSURFFeatures(img1);
    [features6, valid_points6] = extractFeatures(img1, points6, 'Method', 'SURF');
    ext_FT(6,numb_img) = toc;
    ext_F{6,numb_img} = features6;
    ext_FPts{6,numb_img} = valid_points6;

    % 결과 출력(figure)
    figure(1); clf;
    set(gcf, 'Position',  [100, 100, 900, 1200])
    t = tiledlayout(4,2);
    t.Padding = 'none';
    t.TileSpacing = 'compact';
    title(t,'Feature Extraction Results')

    nexttile([1 2])       % subplot(4,2,[1 2])
    imshow(fr1)
    title(sprintf('Original Image (#%d/%d)', numb_img, size(list_img,1)))
    for numb_alg = 1:size(name_alg,2)
        nexttile  % subplot(4,2,(numb_alg+2))
        imshow(img1)
        hold on
        pts=ext_FPts{numb_alg,numb_img};
        pts=pts.Location;
        scatter(pts(:,1), pts(:,2), 3,'+g')
        title(sprintf('%s (%d pts, %0.2f sec)', name_alg{numb_alg}, size(pts,1), ext_FT(numb_alg,numb_img)))

    end
    pause(0.1) %>> 그림이 정상적으로 출력되지 않거나, 출력이 밀리는 경우 방지를 위한 코드 (0.1초 동안 코드 실행 일시정지)


end


%% Feature Matching {'BRISK' 'FAST' 'KAZE' 'ORB' 'SIFT' 'SURF'};
close all
clc
msg_alg={'BRISK' 'FAST ' 'KAZE ' 'ORB  ' 'SIFT   ' 'SURF '};

Mat_F = cell([2+size(name_alg,2), size(image_pair,1)]); % 매칭 결과 저장을 위한 빈 행렬
Mat_F(1:2,:) = num2cell(image_pair'); % Mat_F = [이미지 번호1; 이미지번호2; 알고리즘 1번 결과...]
Mat_FT = NaN(8, size(image_pair,1));   % 특징점 매칭에 걸리는 시간 저장을 위한 빈 행렬
Mat_FT(1:2,:) = image_pair';

for numb_imgp=1:size(image_pair,1)

    numb_img1 = image_pair(numb_imgp,1); % number(index) of image 1
    numb_img2 = image_pair(numb_imgp,2); % number(index) of image 2

    % 이미지 불러오기
    filename1 = list_img{numb_img1,1}; % 이미지 경로/이름
    fr1 = imread(filename1); % 이미지 불러 오기
    fr1 = imresize(fr1, scalef); % 이미지 크기 조정
    img1 = rgb2gray(fr1); % 특징점 추출을 위하여 이미지를 전정색(회색조)으로 변환
    filename2 = list_img{numb_img2,1}; % 이미지 경로/이름
    fr2 = imread(filename2); % 이미지 불러 오기
    fr2 = imresize(fr2, scalef); % 이미지 크기 조정
    img2 = rgb2gray(fr2); % 특징점 추출을 위하여 이미지를 전정색(회색조)으로 변환

    % Show original images(image pair)
    figure(2); clf;
    set(gcf, 'Position',  [100, 100, 900, 1200])
    t = tiledlayout(4,2);
    t.Padding = 'none';
    t.TileSpacing = 'compact';
    title(t,'Feature Matching Results')

    nexttile
    imshow(fr1)
    title(sprintf('Image %d (#%d/%d)', numb_img1, numb_imgp, size(image_pair,1)))
    
    nexttile
    imshow(fr2)
    title(sprintf('Image %d (#%d/%d)', numb_img2, numb_imgp, size(image_pair,1)))

    % Feature matching using {'BRISK' 'FAST' 'KAZE' 'ORB' 'SIFT' 'SURF'};   
    fprintf('\n[ Feature Matching: %d-%d ]\n', numb_img1, numb_img2)   
    for numb_alg = 1:size(msg_alg,2) 

        f1 = ext_F{numb_alg,numb_img1}; % feature of image1
        f2 = ext_F{numb_alg,numb_img2}; % feature of image2
        pts1 = ext_FPts{numb_alg,numb_img1}; % coordinates of features in image1
        pts2 = ext_FPts{numb_alg,numb_img2}; % coordinates of features in image2

        tic
        indexPairs = matchFeatures(f1,f2,'Method','Approximate'); % feature matching, 이미지 특성에 따라서 적절히 변경하여 method를 설정하도록 추후 변경)
        Mat_FT(numb_alg+2,numb_imgp) = toc;        

        fprintf('[ %s ] %d pts (PT=%0.2fs)\n',...
            msg_alg{numb_alg}, size(indexPairs,1), Mat_FT(numb_alg+2,numb_imgp))
        % fprintf('[%d-%d %s] >>> %d pts (PT=%0.2fs, PT/pts=%0.4fms)\n',...
        %     numb_img1, numb_img2, name_alg{numb_alg}, size(indexPairs,1), Mat_FT(numb_alg+2,numb_imgp), Mat_FT(numb_alg+2,numb_imgp)/size(indexPairs,1)*1000)

        matched_Pts1 = pts1(indexPairs(:,1),:);
        matched_Pts2 = pts2(indexPairs(:,2),:);
        Mat_F(numb_alg+2,numb_imgp)= {{indexPairs; matched_Pts1; matched_Pts2}};
        
        % show matching results
        nexttile
        showMatchedFeatures(img1,img2,matched_Pts1,matched_Pts2);
        title(sprintf('%s (%d pts, %0.2f sec)', name_alg{numb_alg}, size(indexPairs,1), Mat_FT(numb_alg+2,numb_imgp)))
        pause(0.1)
    end

    if numb_imgp ~= size(image_pair,1)
        disp('Press any key to continue...')
        pause
    end

end







