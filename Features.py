'''
All features and targets we might want for the dataset
& implementation methods
'''

#list of valid feature string names
valid_features = ["lag_return"]

#list of valid target string names
valid_targets = []

#parent class for features and targets
class Variable:
    def __init__(self, name, feature:bool=True):
        self.name = name
        self.feature = feature

    def get(self, df, num_days):
        """Placeholder function for getting the feature

        Args:
            df (_type_): dataframe to get feature on
            num_days (_type_): # of days to look back

        Returns:
            pd.Series: pandas series of the feature data
        """
        return None
    
class LagReturn(Variable):
    def __init__(self):
        super().__init__("lag_return")

    def get(self, df, num_days):
        self.detailed_name = f"{self.name}_{num_days}d"
        
