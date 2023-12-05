function[grp_labels, grp_effects] = GFE(Y,G,ninit)
[grp_labels, grp_effects] = kmeans(Y,G, 'MaxIter',10000);
end


