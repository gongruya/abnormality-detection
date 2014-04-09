function names = name_filtering(fileName)

    Dirs = dir(fileName);
    names = {};
    kk = 0;
    for ii = 1 : length(Dirs)
        if ~strcmp(Dirs(ii).name(1),'.')
            kk = kk + 1;
            names{kk} = Dirs(ii).name;
        end
    end

end