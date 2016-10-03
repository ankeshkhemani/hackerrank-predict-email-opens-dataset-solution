#! /usr/local/bin/octave -qf

function [X y] = preProcessing(inputfile,task)

#################################################
##Data Preprocessing Steps
#################################################
#Create new relevant attributes using existing attributes
#Create logical output classifications based on existing output attributes
#Convert attributes which are classifiers into binary row vectors 

#In octave#
#Load Input data

load(inputfile);

#Check data features available using 'whos'
#Variables in the current scope:

#   Attr Name                                      Size                     Bytes  Class
#   ==== ====                                      ====                     =====  ===== 
#        contest_login_count                       1x98843                 790744  double
#        contest_login_count_1_days                1x98843                 790744  double
#        contest_login_count_30_days               1x98843                 790744  double
#        contest_login_count_365_days              1x98843                 790744  double
#        contest_login_count_7_days                1x98843                 790744  double
#        contest_participation_count               1x98843                 790744  double
#        contest_participation_count_1_days        1x98843                 790744  double
#        contest_participation_count_30_days       1x98843                 790744  double
#        contest_participation_count_365_days      1x98843                 790744  double
#        contest_participation_count_7_days        1x98843                 790744  double
#        forum_comments_count                      1x98843                 790744  double
#        forum_count                               1x98843                 790744  double
#        forum_expert_count                        1x98843                 790744  double
#        forum_questions_count                     1x98843                 790744  double
#        hacker_confirmation                       1x98843                 790744  double
#        hacker_created_at                         1x98843                 790744  double
#        hacker_timezone                           1x98843                 790744  double
#        ipn_count                                 1x98843                 790744  double
#        ipn_count_1_days                          1x98843                 790744  double
#        ipn_count_30_days                         1x98843                 790744  double
#        ipn_count_365_days                        1x98843                 790744  double
#        ipn_count_7_days                          1x98843                 790744  double
#        ipn_read                                  1x98843                 790744  double
#        ipn_read_1_days                           1x98843                 790744  double
#        ipn_read_30_days                          1x98843                 790744  double
#        ipn_read_365_days                         1x98843                 790744  double
#        ipn_read_7_days                           1x98843                 790744  double
#        last_online                               1x98843                 790744  double
#        mail_category                             1x98843                 790744  double
#        mail_type                                 1x98843                 790744  double
#        opened                                    1x98843                 790744  double
#        sent_time                                 1x98843                 790744  double
#        submissions_count                         1x98843                 790744  double
#        submissions_count_1_days                  1x98843                 790744  double
#        submissions_count_30_days                 1x98843                 790744  double
#        submissions_count_365_days                1x98843                 790744  double
#        submissions_count_7_days                  1x98843                 790744  double
#        submissions_count_contest                 1x98843                 790744  double
#        submissions_count_contest_1_days          1x98843                 790744  double
#        submissions_count_contest_30_days         1x98843                 790744  double
#        submissions_count_contest_365_days        1x98843                 790744  double
#        submissions_count_contest_7_days          1x98843                 790744  double
#        submissions_count_master                  1x98843                 790744  double
#        submissions_count_master_1_days           1x98843                 790744  double
#        submissions_count_master_30_days          1x98843                 790744  double
#        submissions_count_master_365_days         1x98843                 790744  double
#        submissions_count_master_7_days           1x98843                 790744  double
#        unsubscribed                              1x98843                 790744  double

#Total is 4744464 elements using 37955712 bytes

if (task==1)
	#Outcomes to be classified(Optional, tried didn't make a difference): -
	multiclassY=zeros(size(opened));
	#Open and does not unsubscribe
	#o_nu=(1-unsubscribed).*opened;
	#Open and Unsubscribe
	#o_u=opened.*unsubscribed;
	#Does not open and does not unsubscribe
	#no_nu=(1-opened).*(1-unsubscribed);
	#Does not open and unsubscribe
	#no_u=(1-opened).*unsubscribed;
	#Load  classified outcomes in y
	#multiclassY=(o_nu+2*o_u+3*no_nu+4*no_u);
	#y={1,2} means mail was opened. Which is the desired outcome.
	y=opened;
endif


#New features

# Unread Ipn fraction
ipn_unread=500*(ipn_count-ipn_read)./(1+ipn_count);
ipn_unread_1_days=(ipn_count_1_days-ipn_read_1_days)./(1+ipn_count_1_days);
ipn_unread_7_days=500*(ipn_count_7_days-ipn_read_7_days)./(1+ipn_count_7_days);
ipn_unread_30_days=500*(ipn_count_30_days-ipn_read_30_days)./(1+ipn_count_30_days);
ipn_unread_365_days=500*(ipn_count_365_days-ipn_read_365_days)./(1+ipn_count_365_days);

# Read Ipn fraction
ipn_read=500*ipn_read ./ (1+ipn_count);
ipn_read_1_days=500*ipn_read_1_days ./ (1+ipn_count_1_days);
ipn_read_7_days=500*ipn_read_7_days ./ (1+ipn_count_7_days);
ipn_read_30_days=500*ipn_read_30_days ./ (1+ipn_count_30_days);
ipn_read_365_days=500*ipn_read_365_days ./ (1+ipn_count_365_days);

# Time Gap between last online and mail sent(sent_time, last_online)
timegap=sent_time-last_online;

#Change Classifier input into multiclass 1s and 0s

Mail_category=zeros(max(mail_category),length(mail_category));
for i=1:size(Mail_category,1)
    Mail_category(i,:)=(mail_category==i);
end

#Analyse features and Load Chosen features in X

#Following Original Features are likely to be linearly proportional to chance of opening mail.
LinearFeatures=[contest_login_count;contest_login_count_1_days;contest_login_count_30_days;contest_login_count_365_days;contest_login_count_7_days;contest_participation_count;contest_participation_count_1_days;contest_participation_count_30_days;contest_participation_count_365_days;contest_participation_count_7_days;forum_comments_count;forum_count;forum_expert_count;forum_questions_count;hacker_confirmation;submissions_count;submissions_count_1_days;submissions_count_30_days;submissions_count_365_days;submissions_count_7_days;submissions_count_contest;submissions_count_contest_1_days;submissions_count_contest_30_days;submissions_count_contest_365_days;submissions_count_contest_7_days;submissions_count_master;submissions_count_master_1_days;submissions_count_master_30_days];

#Following Modified features are Highly likely to be linearly proportional to chance of opening mail.
ModLinearFeatures=[ipn_read;ipn_read_1_days;ipn_read_30_days;ipn_read_365_days;ipn_read_7_days;ipn_unread;ipn_unread_1_days;ipn_unread_30_days;ipn_unread_365_days;ipn_unread_7_days;timegap];

#Following Modified Classification features are likely to be linearly proportional to chance of opening mail.
ClassificationFeatures=[Mail_category];

X=[ClassificationFeatures;ModLinearFeatures;LinearFeatures];

#Variables to be saved in file, each row for each example
X=X';
LinearFeatures=LinearFeatures';
ModLinearFeatures=ModLinearFeatures';
ClassificationFeatures=ClassificationFeatures';

if (task==1)
    opened=opened';
    multiclassY=multiclassY';
    y=y';
    %save data/relevant_input_data.mat X LinearFeatures ModLinearFeatures ClassificationFeatures y opened multiclassY;
elseif (task==2)
    %save data/relevant_test_data.mat X LinearFeatures ModLinearFeatures ClassificationFeatures;
endif

end