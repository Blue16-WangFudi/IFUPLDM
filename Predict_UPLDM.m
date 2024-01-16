function  pred = Predict_UPLDM(Outs_Train, Samples_Predict)

   deta = Outs_Train.deta; 
   Samples_Train = Outs_Train.Samples;
   Kernel = Outs_Train.Ker;
   
   sum = 0;
   for i = 1:length(Samples_Train)
       sum = sum + deta(i) * Function_Kernel(Samples_Train(i,:), Samples_Predict, Kernel);
   end
   
   % Use the sign function to get the predicted class
   pred = sign(sum);
   
end  


