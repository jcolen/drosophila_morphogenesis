if ~exist('da', 'var')
    buildAtlas;
end

root = '..//Public';
genotype = 'WT';

labels = {...
    'Fushi_Tarazu',...
    'Paired',...
    'Sloppy_Paired',...
    'Tartan', ...
};

for ii = 1:length(labels)
    label = labels{ii}
    
    qs = da.findGenotypeLabel(genotype, label);
    info = qs.meta
    save(fullfile(root, genotype, label, 'static_queried_sample.mat'), '-struct', 'info');

end