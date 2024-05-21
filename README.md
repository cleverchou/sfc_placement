
Test：
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/100s6 --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=100
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/200s6 --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=200
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/300s6 --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=300
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/100s8 --min_length=8 --max_length=8 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=100
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/200s8 --min_length=8 --max_length=8 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=200
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/300s8 --min_length=8 --max_length=8 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=300
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/100s10 --min_length=10 --max_length=10 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=100
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/200s10 --min_length=10 --max_length=10 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=200
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/300s10 --min_length=10 --max_length=10 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=300
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/100s12 --min_length=12 --max_length=12 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=100
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/200s12 --min_length=12 --max_length=12 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=200
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/300s12 --min_length=12 --max_length=12 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=300

python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=300
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=8 --max_length=8 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=300
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=10 --max_length=10 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=300
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=12 --max_length=12 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=300
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=600
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=8 --max_length=8 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=600
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=10 --max_length=10 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=600
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=12 --max_length=12 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=600
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=900
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=8 --max_length=8 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=900
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=10 --max_length=10 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=900
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/ll3b --min_length=12 --max_length=12 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=900

Learn：
python main.py --learn_mode=True --save_model=True --save_to=save/300 --num_epoch=20000 --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=10 --env_profile="small_default" --batch_size=300
python main.py --learn_mode=True --save_model=True --save_to=save/300 --num_epoch=20000 --min_length=6 --max_length=6 --num_layers=4 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --batch_size=900

