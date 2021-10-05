% Motion model
function [ pose ] = f(pose,T,v,w)
    % This function is the observation model.
    % Inputs: 
    %   pose - 3 x 1 vector of state
    %   T - interval between consecutive timestamps
    %   v - 2 x 1 vector containing translational and rotational velocities
    %   w - 2 x 1 vector of velocity measurement noises
    % Output:
    %   y - 2 x 1 vector including range and bearing measurements
    
    % Propagate state forward in time
    theta = pose(3);
    temp = [cos(theta),0;sin(theta),0;0,1];
    
    % Compute pose
    pose = pose + T.*temp*(v+w);
end