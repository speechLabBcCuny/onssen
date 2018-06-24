addpath('/home/near/mimlib/');
addpath('/home/near/utils');
fileBase = '/scratch/near/2speakers/wav8k/min/';
outBase = '/scratch/near/2speakers_stft/wav8k/min/';
inFiles = findFiles(fileBase,'.*\.wav');
for i = 1:length(inFiles)
  fprintf('start processing file: %d\n',i);
  [wave, fs] = audioread(fullfile(fileBase,inFiles{i}));
  stft = stft_multi(wave.',256);
  nsampl = length(wave);
  outFile = fullfile(outBase,replace(inFiles{i},'wav','mat'));
  ensureDirExists(outFile);
  save(outFile, 'stft', 'nsampl', '-v6');
end
