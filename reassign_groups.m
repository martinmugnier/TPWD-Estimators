function new_groups = reassign_groups(group_memberships,G)
    % REASSIGN_GROUPS Randomly reassign units to empty groups so that no 
    % group is empty after Bonhomme and Manresa (2015)'s Algorithm 1.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % group_memberships : Nx1 array of current group memberships;
    % G                 : number of groups.
    %
    % OUTPUT:
    % -------
    % new_groups        : Nx1 array of new group memberships.
    %
    % REFERENCE:
    % ----------
    % Bonhomme, S. and Manresa, E. (2015), Grouped Patterns of 
    % Heterogeneity in Panel Data. Econometrica, 83: 1147-1184.

    % Get the current count of members in each group
    group_counts = histcounts(group_memberships, 1:G+1);

    % Identify the groups with zero members and groups with at least 
    % two members
    zero_member_groups = find(group_counts == 0);
    two_or_more_member_groups = find(group_counts >= 2);

    % Initialize the new group memberships to the current memberships
    new_groups = group_memberships;

    % Reassign one unit from groups with at least two members to zero
    % member groups 
    for i = 1:length(zero_member_groups)
        % Ensure there are still donor groups with at least two members
        if isempty(two_or_more_member_groups)
            error('Not enough groups with at least two members to reassign units.');
        end
        
        % Select a donor group with at least two members
        donor_group = two_or_more_member_groups(1);
        
        % Get the indices of units in the donor group
        donor_indices = find(new_groups == donor_group);
        
        % Randomly select one unit from the donor group
        unit_to_reassign = donor_indices(randi(length(donor_indices)));
        
        % Reassign the selected unit to the current zero member group
        new_groups(unit_to_reassign) = zero_member_groups(i);
        
        % Update group counts
        group_counts(donor_group) = group_counts(donor_group) - 1;
        
        % If the donor group no longer has at least two members, 
        % % remove it from the list
        if group_counts(donor_group) < 2
            two_or_more_member_groups = two_or_more_member_groups(2:end);
        end
    end
end