if ~exist('da', 'var')
    buildAtlas;
end

root = '/Users/jcolen/Documents/drosophila_morphogenesis/flydrive';

labels = {...
    'Sqh-GFP',...
    %'Moesin-GFP',...
    %'ECad-GFP',...
    %'Runt',...
    %'Even_Skipped-YFP',...
    %'histone-RFP',...
};

genotype = 'WT';
%genotype = 'toll[RM9]';
%genotype = 'spaetzle[A]';
%genotype = 'Halo_twist[ey53]';
genotype = 'Halo_Hetero_Twist[ey53]_Hetero'

for ii = 1:length(labels)
    label = labels{ii}
    
    qs = da.findDynamicGenotypeLabel(genotype, label);
    info = qs.meta
    save(fullfile(root, genotype, label, 'dynamic_queried_sample.mat'), '-struct', 'info');
    
    options = struct;
    options.overwrite = false;
    options.method = 'default';
    qs.ensurePIV(options);
    qs.getPIV(options);

end