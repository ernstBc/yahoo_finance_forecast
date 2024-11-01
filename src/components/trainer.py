import datetime
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from src.config import MODEL_DIR, FORECAST_DIR
from src.utils import save_artifact
from dataclasses import dataclass

@dataclass
class TrainerArimaConfig:
    model_id=datetime.datetime.now().strftime('%Y_%m_%d')
    model_path=MODEL_DIR
    model_save_dir=Path.joinpath(model_path, f'{model_id}.pkl')
    forecast_save_dir=Path.joinpath(FORECAST_DIR, f'{model_id}.pkl')



class TrainerARIMA:
    def __init__(self):
        self.config=TrainerArimaConfig()

    def init_trainer(self, train_dataset, max_ar:int, max_i:int,max_ma:int):
        model, forecast, _=search_model(train_dataset=train_dataset, max_ar=max_ar, max_i=max_i, max_ma=max_ma)
        print('path save', str(self.config.model_path)+str(self.config.model_id)+'.plk')

        save_artifact(self.config.model_save_dir, model)
        save_artifact(self.config.forecast_save_dir, forecast)

        return self.config.model_save_dir, self.config.forecast_save_dir




def search_model(train_dataset, max_ar:int, max_i:int, max_ma:int):
    best_arima=(0,0,0)
    best_aic=None
    report=None
    for i in range(max_i+1):
        for ar in range(max_ar+1):
            for ma in range(max_ma+1):
                model=ARIMA(train_dataset, order=(ar, i, ma))
                res=model.fit()
                aic_model=res.aic

                if best_aic is None:
                    best_aic=aic_model
                    best_arima=(ar,i,ma)
                    report=res
                else:
                    if aic_model < best_aic:
                        best_arima=(ar,i,ma)
                        best_aic=aic_model
                        report=res
    return model, report, best_arima


if __name__=='__main__':
    traif=TrainerARIMA()
    print(traif.config.model_save_dir)