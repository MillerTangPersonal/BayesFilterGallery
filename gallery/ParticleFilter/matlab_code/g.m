function [ y ] = g(pose,l,d,n)
    % This function is the observation model.
    % Inputs: 
    %   pose - 3 x 1 vector of state
    %   l - 2 x 1 vector of landmark position
    %   d - distance between laser range sensor and centre of robot
    %   n - 2 x 1 vector of range and bearing measurement noises
    % Output:
    %   y - 2 x 1 vector including range and bearing measurements
    
    % Compute range and bearing
    alpha = l(1) - pose(1) - d*cos(pose(3));
    beta = l(2) - pose(2) - d*sin(pose(3));
    range = sqrt(alpha^2 + beta^2) + n(1);
    bearing = atan2(beta,alpha) - pose(3) + n(2);
    
    % Restrict bearing to -pi and pi
    while bearing > pi
        bearing = bearing - 2*pi;
    end
	while bearing < -pi
        bearing = bearing + 2*pi;
    end
    
    % Stack observation vector
    y = [range;bearing];
end