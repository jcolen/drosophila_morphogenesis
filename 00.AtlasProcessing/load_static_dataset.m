if ~exist('da', 'var')
    buildAtlas;
end

root = '../flydrive';
genotype = 'WT';

labels = {...
    'Fushi_Tarazu',...
    'Paired',...
    'Sloppy_Paired',...
    'Even_Skipped', ...
    'Runt', ...
    'Hairy', ...
};

for ii = 1:length(labels)
    label = labels{ii}
    
    qs = da.findGenotypeLabel(genotype, label);
    info = qs.meta
    save(fullfile(root, genotype, label, 'static_queried_sample.mat'), '-struct', 'info');

end