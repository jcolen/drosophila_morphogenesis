clc;
clear;

%% Add paths (this part can be slow)
cd /Users/jcolen/Documents/drosophila_morphogenesis/00.AtlasProcessing;

addpath(genpath('../flydrive/dynamicAtlas'));
addpath(genpath('../flydrive/dynamicAtlas/+dynamicAtlas'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define atlasPath to be where the dynamicAtlas resides (the parent
% dynamicAtlas directory, not the project directory '+dynamicAtlas')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
atlasPath = '/Users/jcolen/Documents/drosophila_morphogenesis/flydrive';
da = BuildAtlas(atlasPath);

function [da] = BuildAtlas(atlasPath)
    %% Base Atlas
    genotypes = {'WT', 'Halo_Hetero_Twist[ey53]_Hetero'};
    options = struct();
    options.labels = {};
    options.timerfn = {...
        'timematch_linearOffset_vmagAlign.txt', ...
        'timematch_linearOffset_curlSign.txt', ...
        'timematch_*_*stripe7_chisq.mat', ...
        'timematch_*_cephallicFurrowOnset.txt',...
        };
    da = dynamicAtlas.dynamicAtlas(atlasPath, genotypes, options);

    %% Mutant atlas (collated Toll/Spatzle)
    genotypes = {'toll[RM9]', 'spaetzle[A]'};
    options.prependFileName = 'cylinder1_*max';
    options.labels = {'Sqh-GFP'};
    da1 = dynamicAtlas.dynamicAtlas(atlasPath, genotypes, options);
    da = mergeAtlas(da, da1);

    %% Mutant atlas (Twist mutants)
    genotypes = {'Halo_twist[ey53]'};
    options.prependFileName = '202*';
    options.labels = {'Sqh-GFP'};
    da1 = dynamicAtlas.dynamicAtlas(atlasPath, genotypes, options) ;
    da = mergeAtlas(da, da1);

end

function [da1] = mergeAtlas(da1, da2)
    da1.genotypes = [da1.genotypes, da2.genotypes];
    for ii = 1:numel(da2.genotypes)
        key = da2.genotypes{ii};
        da1.lookup(key) = da2.lookup(key);
    end
end